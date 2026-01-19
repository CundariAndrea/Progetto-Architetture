#include <stdlib.h>
#include <stdio.h>
#include <xmmintrin.h>
#include <math.h>
#include <omp.h>
#include "common.h"

extern void prova(params* input);

void fit(params* input){
    // Selezione dei pivot
    // Costruzione dell'indice
    input->index = _mm_malloc(8*sizeof(type), align);
}

void predict(params* input){
    // Esecuzione delle query
    input->id_nn[1] = 5;
    prova(input);
}
