#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "results_check.h"

// --- Helper interno per caricare header ---
static int read_header(FILE* fp, int* rows, int* cols) {
    if (fread(rows, sizeof(int), 1, fp) != 1) return 0;
    if (fread(cols, sizeof(int), 1, fp) != 1) return 0;
    return 1;
}

int compare_id_files(const char* file_ref, const char* file_test) {
    FILE *f1 = fopen(file_ref, "rb");
    FILE *f2 = fopen(file_test, "rb");

    if (!f1 || !f2) {
        fprintf(stderr, "Test FAIL: Impossibile aprire i file ID.\n");
        if(f1) fclose(f1);
        if(f2) fclose(f2);
        return -1;
    }

    int r1, c1, r2, c2;
    if (!read_header(f1, &r1, &c1) || !read_header(f2, &r2, &c2)) {
        fprintf(stderr, "Test FAIL: Errore lettura header ID.\n");
        fclose(f1); fclose(f2);
        return -1;
    }

    if (r1 != r2 || c1 != c2) {
        fprintf(stderr, "Test FAIL: Dimensioni diverse ID (Ref: %dx%d vs Test: %dx%d)\n", r1, c1, r2, c2);
        fclose(f1); fclose(f2);
        return -1;
    }

    size_t num_elems = (size_t)r1 * c1;
    int* data1 = (int*)malloc(num_elems * sizeof(int));
    int* data2 = (int*)malloc(num_elems * sizeof(int));

    if (!data1 || !data2) {
        fprintf(stderr, "Test FAIL: Errore malloc ID.\n");
        free(data1); free(data2);
        fclose(f1); fclose(f2);
        return -1;
    }

    fread(data1, sizeof(int), num_elems, f1);
    fread(data2, sizeof(int), num_elems, f2);

    int errors = 0;
    for (size_t i = 0; i < num_elems; i++) {
        if (data1[i] != data2[i]) {
            errors++;
        }
    }

    if (errors == 0) {
        printf("Test OK (ID)\n");
    } else {
        printf("Test FAIL: Trovati %d mismatch negli ID.\n", errors);
    }

    free(data1);
    free(data2);
    fclose(f1);
    fclose(f2);

    return errors;
}

int compare_dist_files_internal(const char* file_ref, const char* file_test, size_t elem_size, double epsilon) {
    FILE *f1 = fopen(file_ref, "rb");
    FILE *f2 = fopen(file_test, "rb");

    if (!f1 || !f2) {
        fprintf(stderr, "Test FAIL: Impossibile aprire i file Distanze.\n");
        if(f1) fclose(f1);
        if(f2) fclose(f2);
        return -1;
    }

    int r1, c1, r2, c2;
    if (!read_header(f1, &r1, &c1) || !read_header(f2, &r2, &c2)) {
        fprintf(stderr, "Test FAIL: Errore lettura header Distanze.\n");
        fclose(f1); fclose(f2);
        return -1;
    }
    if (r1 != r2 || c1 != c2) {
        fprintf(stderr, "Test FAIL: Dimensioni diverse Distanze (Ref: %dx%d vs Test: %dx%d)\n", r1, c1, r2, c2);
        fclose(f1); fclose(f2);
        return -1;
    }

    // Controllo validità elem_size
    int is_float = (elem_size == sizeof(float));
    int is_double = (elem_size == sizeof(double));
    if (!is_float && !is_double) {
        fprintf(stderr, "Test FAIL: Dimensione elemento non supportata (%zu bytes). Atteso %zu (float) o %zu (double).\n", 
                elem_size, sizeof(float), sizeof(double));
        fclose(f1); fclose(f2);
        return -1;
    }

    size_t num_elems = (size_t)r1 * c1;

    // Allocazione buffer con dimensione esatta passata via parametro
    void* raw1 = malloc(num_elems * elem_size);
    void* raw2 = malloc(num_elems * elem_size);
    if (!raw1 || !raw2) {
        fprintf(stderr, "Test FAIL: Errore malloc Distanze.\n");
        free(raw1); free(raw2);
        fclose(f1); fclose(f2);
        return -1;
    }
    fread(raw1, elem_size, num_elems, f1);
    fread(raw2, elem_size, num_elems, f2);

    int errors = 0;
    double max_diff = 0.0;

    for (size_t i = 0; i < num_elems; i++) {
        double val1, val2;

        if (is_float) {
            val1 = (double)((float*)raw1)[i];
            val2 = (double)((float*)raw2)[i];
        } else {
            val1 = ((double*)raw1)[i];
            val2 = ((double*)raw2)[i];
        }
        double diff = fabs(val1 - val2);
        if (diff > max_diff) {
            max_diff = diff;
        }
        if (diff > epsilon) {
            errors++;
        }
    }

    if (errors == 0) {
        printf("Test OK (Dist) - Max Diff: %g\n", max_diff);
    } else {
        printf("Test FAIL: Trovati %d mismatch Distanze (Max Diff: %g)\n", errors, max_diff);
    }

    free(raw1);
    free(raw2);
    fclose(f1);
    fclose(f2);

    return errors;
}
