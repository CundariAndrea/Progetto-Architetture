#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <xmmintrin.h>
#include <omp.h>        
#include "common.h"

// --- ATTENZIONE: NON INCLUDERE FILE .C QUI! ---

// Dichiarazione funzioni esterne
extern void fit(params* input);
extern void predict(params* input);

// Helper per errori
static void die(const char* msg) {
    fprintf(stderr, "Errore: %s\n", msg);
    exit(EXIT_FAILURE);
}

// Helper matematico
static inline type euclid_dist(const type* a, const type* b, int D) {
    double acc = 0.0;
    for (int i = 0; i < D; i++) {
        double d = (double)a[i] - (double)b[i];
        acc += d * d;
    }
    return (type)sqrt(acc);
}

// Verifica coerenza
static void check_dist_nn_consistency(const params* p, int qi, double eps) {
    const type* q = &p->Q[(size_t)qi * (size_t)p->D];
    const int* ids = &p->id_nn[(size_t)qi * (size_t)p->k];
    const type* dists = &p->dist_nn[(size_t)qi * (size_t)p->k];
 
    for (int i = 0; i < p->k; i++) {
        int id = ids[i];
        if (id < 0 || id >= p->N) continue;
        const type* v = &p->DS[(size_t)id * (size_t)p->D];
        type d = euclid_dist(q, v, p->D);
        double diff = fabs((double)d - (double)dists[i]);
 
        if (diff > eps) {
            printf("[FAIL] qi=%d pos=%d id=%d dist_nn=%f euclid=%f diff=%g\n",
                   qi, i, id, (double)dists[i], (double)d, diff);
        }
    }
}

// Loader
MATRIX load_data(const char* filename, int *n, int *d) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) die("Impossibile aprire file dati");
 
    int rows = 0, cols = 0;
    if (fread(&rows, sizeof(int), 1, fp) != 1) die("Errore header righe");
    if (fread(&cols, sizeof(int), 1, fp) != 1) die("Errore header colonne");
 
    MATRIX data = (MATRIX)_mm_malloc((size_t)rows * (size_t)cols * sizeof(type), 16);
    if (!data) die("Malloc fallita");
 
    size_t tot = (size_t)rows * (size_t)cols;
    if (fread(data, sizeof(type), tot, fp) != tot) die("Errore lettura corpo dati");
    fclose(fp);
    *n = rows; *d = cols;
    return data;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Uso: %s <dataset> <query> [N D k]\n", argv[0]);
        return 1;
    }
 
    params p = {0};
    printf("Caricamento dataset...\n");
    p.DS = load_data(argv[1], &p.N, &p.D);
    printf("Caricamento query...\n");
    p.Q  = load_data(argv[2], &p.nq, &p.D);
 
    p.h = 8;
    p.k = 5;
    p.x = 4;
    p.silent = 0;
    
    if (argc >= 6) p.k = atoi(argv[5]);

    printf("Start Fit...\n");
    double t0 = omp_get_wtime();
    fit(&p);
    double t1 = omp_get_wtime();
    printf("Fit completata in %f sec\n", t1 - t0);
 
    printf("Start Predict...\n");
    t0 = omp_get_wtime();
    predict(&p);
    t1 = omp_get_wtime();
    printf("Predict completata in %f sec\n", t1 - t0);

    check_dist_nn_consistency(&p, 0, 1e-4);
    
    printf("\nKNN Query 0:\n");
    for (int i = 0; i < p.k; i++) {
        printf(" %d) id=%d dist=%f\n", i+1, p.id_nn[i], p.dist_nn[i]);
    }
 
    _mm_free(p.DS); _mm_free(p.Q);
    if(p.index) _mm_free(p.index);
    if(p.dist_nn) _mm_free(p.dist_nn);
    free(p.P); free(p.id_nn);
 
    return 0;
}