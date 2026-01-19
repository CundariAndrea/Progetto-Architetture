#include <stdio.h>
#include <stdlib.h>
#include <xmmintrin.h>
#include "data_io.h"

void* load_data_internal(const char* filename, int *rows, int *cols, size_t elem_size, int alignment) {
    FILE* fp;
    int r, c;

    fp = fopen(filename, "rb");
    if (fp == NULL){
        fprintf(stderr, "Errore: impossibile aprire il file '%s'!\n", filename);
        exit(EXIT_FAILURE);
    }

    fread(&r, sizeof(int), 1, fp);
    fread(&c, sizeof(int), 1, fp);

    void* data = _mm_malloc(r * c * elem_size, alignment);
    if (data == NULL) {
        fprintf(stderr, "Errore: allocazione memoria fallita\n");
        exit(EXIT_FAILURE);
    }

    fread(data, elem_size, r * c, fp);
    fclose(fp);

    *rows = r;
    *cols = c;

    return data;
}

void save_matrix_internal(const char* filename, void* data, int rows, int cols, size_t elem_size) {
    FILE* fp;
    int i;
    char* data_ptr = (char*)data; // Evita warning per l'uso di aritmetica dei puntatori su void*
    size_t row_width_bytes = elem_size * cols;

    fp = fopen(filename, "wb");
    if (!fp) return;

    if(data != NULL){
        fwrite(&rows, sizeof(int), 1, fp);
        fwrite(&cols, sizeof(int), 1, fp);
        for (i = 0; i < rows; i++) {
            fwrite(data_ptr, elem_size, cols, fp);
            data_ptr += row_width_bytes;
        }
    } else {
        int zero = 0;
        fwrite(&zero, sizeof(int), 1, fp);
        fwrite(&zero, sizeof(int), 1, fp);
    }
    fclose(fp);
}

void save_int_matrix_internal(const char* filename, int* data, int rows, int cols) {
    FILE* fp;
    int i;

    fp = fopen(filename, "wb");
    if (!fp) return;

    if(data != NULL){
        fwrite(&rows, sizeof(int), 1, fp);
        fwrite(&cols, sizeof(int), 1, fp);
        for (i = 0; i < rows; i++) {
            fwrite(data, sizeof(int), cols, fp);
            data += cols; 
        }
    } else {
        int zero = 0;
        fwrite(&zero, sizeof(int), 1, fp);
        fwrite(&zero, sizeof(int), 1, fp);
    }
    fclose(fp);
}

void print_matrix_internal(void* data, int rows, int cols, size_t elem_size, const char* label) {
    int i, j;
    // Riconoscimento tipo in base alla dimensione
    int is_float = (elem_size == sizeof(float));
    int is_double = (elem_size == sizeof(double));
    if (!is_float && !is_double) {
        fprintf(stderr, "Errore: tipo di dato non supportato per la stampa (size: %zu)\n", elem_size);
        return;
    }

    // Usiamo char* per muoverci byte per byte nei dati generici
    char* ptr = (char*)data;

    for(i=0; i<rows; i++){
        // Stampa label (es: "Dist NN Q") e indice riga
        printf("%s%3i: ( ", label, i);
        for(j=0; j<cols; j++){
            if(is_float) {
                float val = *(float*)(ptr + (i*cols + j)*elem_size);
                printf("%f ", val);
            } else {
                double val = *(double*)(ptr + (i*cols + j)*elem_size);
                printf("%f ", val);
            }
        }
        printf(")\n");
    }
}

void print_int_matrix_internal(int* data, int rows, int cols, const char* label) {
    int i, j;
    for(i=0; i<rows; i++){
        // Stampa label (es: "Dist NN Q") e indice riga
        printf("%s%3i: ( ", label, i);
        for(j=0; j<cols; j++){
            printf("%i ", data[i*cols + j]);
        }
        printf(")\n");
    }
}
