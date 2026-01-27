#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <xmmintrin.h>
#include <omp.h>

#include "common.h"
#include "../helper/config.h"
#include "../helper/data_io.h"
#include "../helper/results_check.h"
#include "quantpivot64.c"


int main(int argc, char** argv) {
    Config config;
    if (parse_arguments(argc, argv, &config) != 0) {
        exit(EXIT_FAILURE);
    }
    if (!config.silent) print_config(&config);

    params* input = malloc(sizeof(params));
    if (input == NULL) {
        exit(EXIT_FAILURE);
    }
    // 2. Caricamento Dati usando le MACRO (gestiscono sizeof e align automaticamente)
    // Nota: Cast a (MATRIX) necessario perché LOAD_DATA ritorna void*
    input->DS = (MATRIX) LOAD_DATA(config.dsfilename, &input->N, &input->D);
    // Per la query, carichiamo nq (numero query) e sovrascriviamo D (che deve essere uguale)
    input->Q  = (MATRIX) LOAD_DATA(config.queryfilename, &input->nq, &input->D);
    // Allocazione memoria risultati
    input->id_nn   = _mm_malloc(input->nq * config.k * sizeof(int), align);
    input->dist_nn = _mm_malloc(input->nq * config.k * sizeof(type), align);
    // Assegnazione parametri da config a input
    input->h = config.h;
    input->k = config.k;
    input->x = config.x;
    input->silent = config.silent;

    // =========================================================
    // Algoritmo (Fit & Predict)
    // =========================================================

    double t_start, t_end;
    float time_taken;

    // FIT
    t_start = omp_get_wtime(); // Restituisce secondi in doppia precisione
    fit(input);
    t_end = omp_get_wtime();
    time_taken = (float)(t_end - t_start); // Differenza diretta in secondi

    if(!input->silent) printf("FIT time = %.5f secs\n", time_taken);
    else               printf("%.3f\n", time_taken);

    // PREDICT
    t_start = omp_get_wtime();
    predict(input);
    t_end = omp_get_wtime();
    time_taken = (float)(t_end - t_start);

    if(!input->silent) printf("PREDICT time = %.5f secs\n", time_taken);
    else               printf("%.3f\n", time_taken);


    SAVE_INT_MATRIX(config.out_idnn_filename, input->id_nn, input->nq, input->k);
    SAVE_MATRIX(config.out_distnn_filename, input->dist_nn, input->nq, input->k);

    // if(!input->silent){
    //     PRINT_INT_MATRIX("ID NN Q", input->id_nn, input->nq, input->k);
    //     PRINT_MATRIX("Dist NN Q", input->dist_nn, input->nq, input->k);
    // }

    if (!input->silent) {
        printf("\n--- Verifica Output ID ---\n");
    }
    compare_id_files(config.test_resid_filename, config.out_idnn_filename);

    if (!input->silent) {
        printf("\n--- Verifica Output Distanze ---\n");
    }
    COMPARE_DIST_FILES(config.test_resdst_filename, config.out_distnn_filename, 0.0001);

    _mm_free(input->DS);
    _mm_free(input->Q);
    _mm_free(input->P);
    _mm_free(input->index);
    _mm_free(input->id_nn);
    _mm_free(input->dist_nn);
    free(input);

    return 0;
}
