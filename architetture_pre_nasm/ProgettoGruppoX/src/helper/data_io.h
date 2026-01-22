#ifndef DATA_IO_H
#define DATA_IO_H

#include <stddef.h> // Per size_t

// Queste sono le funzioni reali. Sono generiche per funzionare
// sia con float che con double senza duplicare codice.

void* load_data_internal(const char* filename, int *rows, int *cols, size_t elem_size, int alignment);
void save_matrix_internal(const char* filename, void* data, int rows, int cols, size_t elem_size);
void save_int_matrix_internal(const char* filename, int* data, int rows, int cols); // Questa rimane semplice

void print_matrix_internal(void* data, int rows, int cols, size_t elem_size, const char* label);
void print_int_matrix_internal(int* data, int rows, int cols, const char* label);

//  SEZIONE "SEMPLIFICATA" (Per l'uso nel Main)
/**
 * Utilizzo: MATRIX ds = LOAD_DATA("file.ds2", &n, &k);
 */
#define LOAD_DATA(filename, ptr_n, ptr_k) \
    load_data_internal(filename, ptr_n, ptr_k, sizeof(type), align)
/**
 * Utilizzo: SAVE_MATRIX("out.ds2", dataset, n, k);
 */
#define SAVE_MATRIX(filename, data, n, k) \
    save_matrix_internal(filename, data, n, k, sizeof(type))
/**
 * Utilizzo: SAVE_INT_MATRIX("out_id.ds2", id_nn, n, k);
 */
#define SAVE_INT_MATRIX(filename, data, n, k) \
    save_int_matrix_internal(filename, data, n, k)

/**
 * Utilizzo: PRINT_MATRIX("Dist NN Q", input->dist_nn, input->nq, input->k);
 */
#define PRINT_MATRIX(label, data, n, k) \
    print_matrix_internal(data, n, k, sizeof(type), label)

/**
 * Utilizzo: PRINT_INT_MATRIX("ID NN Q", input->id_nn, input->nq, input->k);
 */
#define PRINT_INT_MATRIX(label, data, n, k) \
    print_int_matrix_internal(data, n, k, label)

#endif
