#include <stdio.h>
#include <stdlib.h>
#include <xmmintrin.h>
#include "common.h"
 
// prototipi (se non hai un .h dedicato)
void fit(params* input);
void predict(params* input);

// ---------- utility: error ----------
static void die(const char* msg) {
    fprintf(stderr, "Errore: %s\n", msg);
    exit(EXIT_FAILURE);
}


static void check_dist_nn_consistency(const params* p, int qi, double eps) {
    const type* q = &p->Q[(size_t)qi * (size_t)p->D];
    const int* ids = &p->id_nn[(size_t)qi * (size_t)p->k];
    const type* dists = &p->dist_nn[(size_t)qi * (size_t)p->k];
 
    for (int i = 0; i < p->k; i++) {
        int id = ids[i];
        if (id < 0 || id >= p->N) {
            printf("[FAIL] qi=%d pos=%d id=%d (fuori range)\n", qi, i, id);
            continue;
        }
        const type* v = &p->DS[(size_t)id * (size_t)p->D];
        type d = euclid_dist(q, v, p->D);
        double diff = fabs((double)d - (double)dists[i]);
 
        if (diff > eps) {
            printf("[FAIL] qi=%d pos=%d id=%d dist_nn=%f euclid=%f diff=%g\n",
                   qi, i, id, (double)dists[i], (double)d, diff);
        }
    }
}
 
// loader dataset/query
MATRIX load_data(const char* filename, int *n, int *d) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("'%s': bad data file name!\n", filename);
        exit(EXIT_FAILURE);
    }
 
    int rows = 0, cols = 0;
 
    if (fread(&rows, sizeof(int), 1, fp) != 1)
        die("Errore lettura numero righe");
 
    if (fread(&cols, sizeof(int), 1, fp) != 1)
        die("Errore lettura numero colonne");
 
    MATRIX data = (MATRIX)_mm_malloc((size_t)rows * (size_t)cols * sizeof(type), align);
    if (!data) die("_mm_malloc fallita");
 
    size_t tot = (size_t)rows * (size_t)cols;
    if (fread(data, sizeof(type), tot, fp) != tot)
        die("Errore lettura dati matrice");
 
    fclose(fp);
 
    *n = rows;
    *d = cols;
    return data;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Uso: %s <dataset> <query>\n", argv[0]);
        return 1;
    }
 
    params p = {0};
 
    // dataset
    p.DS = load_data(argv[1], &p.N, &p.D);
 
    // query
    p.Q  = load_data(argv[2], &p.nq, &p.D);
 
    // parametri
    p.h = 8;
    p.k = 5;
    p.x = 4;
    p.silent = 0;
 
    printf("Chiamo fit...\n");
    fit(&p);
    printf("fit OK\n");
 
    printf("Chiamo predict...\n");
    predict(&p);
    printf("predict OK\n");
    check_dist_nn_consistency(&p,0,1e-4);
    
 
    printf("\nKNN prima query:\n");
    for (int i = 0; i < p.k; i++) {
        printf("id=%d dist=%f\n", p.id_nn[i], p.dist_nn[i]);
    }
 
    // cleanup
    _mm_free(p.DS);
    _mm_free(p.Q);
    _mm_free(p.index);
    _mm_free(p.dist_nn);
    free(p.P);
    free(p.id_nn);
 
    return 0;
}


