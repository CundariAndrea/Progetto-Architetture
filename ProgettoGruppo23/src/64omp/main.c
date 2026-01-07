#include <stdio.h>
#include <stdlib.h>
#include <xmmintrin.h>
#include <omp.h>

#include "common.h"

// NON includere .c se puoi evitarlo: meglio compilare separato
#include "quantpivot64omp.c"

MATRIX load_data(char* filename, int *n, int *k) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("'%s': bad data file name!\n", filename);
        exit(1);
    }

    int rows, cols;
    if (fread(&rows, sizeof(int), 1, fp) != 1) exit(1);
    if (fread(&cols, sizeof(int), 1, fp) != 1) exit(1);

    MATRIX data = (MATRIX)_mm_malloc((size_t)rows * (size_t)cols * sizeof(type), align);
    if (!data) exit(1);

    if (fread(data, sizeof(type), (size_t)rows * (size_t)cols, fp) != (size_t)rows * (size_t)cols) exit(1);
    fclose(fp);

    *n = rows;
    *k = cols;
    return data;
}

int main(void) {
    char* dsfilename =
        "/home/christian28/Desktop/Programma da Studiare Esami/Architetture avanzate dei sistemi di elaborazione e programmazione/Progetto-Architetture-2025/dataset_2000x256_64.ds2";
    char* queryfilename =
        "/home/christian28/Desktop/Programma da Studiare Esami/Architetture avanzate dei sistemi di elaborazione e programmazione/Progetto-Architetture-2025/query_2000x256_64.ds2";

    int h = 2;
    int k = 3;
    int x = 2;
    int silent = 0;

    // IMPORTANTISSIMO: azzera la struct
    params* input = (params*)calloc(1, sizeof(params));
    if (!input) return 1;

    input->DS = load_data(dsfilename, &input->N, &input->D);
    input->Q  = load_data(queryfilename, &input->nq, &input->D);

    input->h = h;
    input->k = k;
    input->x = x;
    input->silent = silent;

    double t0, t1;

    t0 = omp_get_wtime();
    fit(input);
    t1 = omp_get_wtime();
    printf("FIT time = %.5f secs\n", (t1 - t0));

    t0 = omp_get_wtime();
    predict(input);
    t1 = omp_get_wtime();
    printf("PREDICT time = %.5f secs\n", (t1 - t0));

    // stampa veloce
    if (!input->silent) {
        printf("Prima query:\n");
        for (int j = 0; j < input->k; j++) {
            printf("  id=%d dist=%lf\n", input->id_nn[j], (double)input->dist_nn[j]);
        }
    }

    // cleanup (ATTENZIONE: P è malloc, non _mm_malloc)
    _mm_free(input->DS);
    _mm_free(input->Q);
    free(input->P);
    _mm_free(input->index);
    free(input->id_nn);
    _mm_free(input->dist_nn);
    free(input);

    return 0;
}