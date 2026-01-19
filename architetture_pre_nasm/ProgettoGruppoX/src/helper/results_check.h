#ifndef RESULT_CHECK_H
#define RESULT_CHECK_H

#include <stddef.h> // Per size_t

/**
 * Confronta due file contenenti matrici di INTERI (ID dei vicini).
 * Formato file: [rows(int)] [cols(int)] [data(int...)]
 * * @param file_ref   Path del file di riferimento
 * @param file_test  Path del file da testare
 * @return           0 se IDENTICI, -1 se errore, >0 numero di mismatch.
 */
int compare_id_files(const char* file_ref, const char* file_test);

/**
 * Funzione interna per confrontare matrici di distanze.
 */
int compare_dist_files_internal(const char* file_ref, const char* file_test, size_t elem_size, double epsilon);

/**
 * Utilizzo: COMPARE_DIST_FILES("ref.ds2", "test.ds2", 0.0001);
 */
#define COMPARE_DIST_FILES(file_ref, file_test, epsilon) \
    compare_dist_files_internal(file_ref, file_test, sizeof(type), epsilon)

#endif
