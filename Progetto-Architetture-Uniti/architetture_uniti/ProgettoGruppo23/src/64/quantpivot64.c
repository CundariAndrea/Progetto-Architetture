#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <xmmintrin.h>
#include <string.h>
#include "common.h"
#include <stdio.h>

void* get_block(int size, int elements) {
    return _mm_malloc((size_t)size * elements, 32);
}

void free_block(void* p) {
    if (p != NULL) {
        _mm_free(p);
    }
}

// =========================================================
// DEFINIZIONE KERNEL
// =========================================================

#ifdef USE_ASM_IMPL
    // Versioni Assembly (definite in quantpivot64asm.nasm)
    extern void quantize_vector(const type* vec, int D, int x, int* out_idx, int* out_sign);
    extern type dist_approx(int* v_idx, int* v_sign, int* w_idx, int* w_sign, int x);
    extern type euclidean_distance(type* v, type* w, int D);

#else
    // Versioni C Statiche (Fallback)
    
    static void quantize_vector(const type* vec, int D, int x, int* out_idx, int* out_sign) {
        for (int i = 0; i < x; i++) out_idx[i] = -1;

        for (int i = 0; i < D; i++) {
            type val = vec[i];
            type abs_val = fabs(val);
            int pos = -1;
            
            for(int j=0; j<x; j++) {
                 if (out_idx[j] == -1 || abs_val > fabs(vec[out_idx[j]])) {
                    pos = j;
                    break;
                }
            }

            if (pos != -1) {
                for (int k = x - 1; k > pos; k--) {
                    out_idx[k] = out_idx[k-1];
                    out_sign[k] = out_sign[k-1];
                }
                out_idx[pos] = i;
                out_sign[pos] = (val >= 0) ? 1 : -1;
            }
        }
    }

    static type dist_approx(int* v_idx, int* v_sign, int* w_idx, int* w_sign, int x) {
        type dot_vp_wp = 0.0;
        type dot_vm_wm = 0.0;
        type dot_vp_wm = 0.0;
        type dot_vm_wp = 0.0;

        for (int i = 0; i < x; i++) {
            for (int j = 0; j < x; j++) {
                if (v_idx[i] == w_idx[j]) {
                    int is_v_plus  = (v_sign[i] == 1);
                    int is_v_minus = (v_sign[i] == 0); 
                    int is_w_plus  = (w_sign[j] == 1);
                    int is_w_minus = (w_sign[j] == 0);

                    if (is_v_plus && is_w_plus) dot_vp_wp += 1.0;
                    else if (is_v_minus && is_w_minus) dot_vm_wm += 1.0;
                    else if (is_v_plus && is_w_minus) dot_vp_wm += 1.0;
                    else if (is_v_minus && is_w_plus) dot_vm_wp += 1.0;
                }
            }
        }
        return dot_vp_wp + dot_vm_wm - dot_vp_wm - dot_vm_wp;
    }

    static type euclidean_distance(type* v, type* w, int D) {
        type sum = 0.0;
        for (int i = 0; i < D; i++) {
            type diff = v[i] - w[i];
            sum += diff * diff;
        }
        return sqrt(sum);
    }
#endif


// =========================================================
// LOGICA FIT & PREDICT (Sempre in C)
// =========================================================

void fit(params* input) {
    int N = input->N;
    int D = input->D;
    int h = input->h;
    int x = input->x;

    input->P = (int*) malloc(h * sizeof(int));
    int step = N / h;
    for (int j = 0; j < h; j++) {
        input->P[j] = step * j;
    }

    input->index = (MATRIX) _mm_malloc(N * h * sizeof(type), 32);

    int* pivot_idxs = malloc(h * x * sizeof(int));
    int* pivot_signs = malloc(h * x * sizeof(int));
    int* point_idx = malloc(x * sizeof(int));
    int* point_sign = malloc(x * sizeof(int));

    for (int j = 0; j < h; j++) {
        quantize_vector(&input->DS[input->P[j] * D], D, x, &pivot_idxs[j*x], &pivot_signs[j*x]);
    }

    for (int i = 0; i < N; i++) {
        quantize_vector(&input->DS[i * D], D, x, point_idx, point_sign);

        for (int j = 0; j < h; j++) {
            input->index[i * h + j] = dist_approx(point_idx, point_sign, &pivot_idxs[j*x], &pivot_signs[j*x], x);
        }
    }

    free(pivot_idxs);
    free(pivot_signs);
    free(point_idx);
    free(point_sign);
}
void predict(params* input) {
    printf("DEBUG: Entrato in predict\n"); fflush(stdout);

    // --- CONTROLLI DI SICUREZZA ---
    if (input == NULL) {
        printf("FATAL ERROR: Struct 'input' is NULL!\n"); fflush(stdout); exit(1);
    }
    printf("DEBUG: Check pointers -> DS=%p, Q=%p, P=%p, index=%p\n", 
           (void*)input->DS, (void*)input->Q, (void*)input->P, (void*)input->index); fflush(stdout);

    if (input->DS == NULL) { printf("FATAL ERROR: input->DS is NULL (Dataset persa?)\n"); exit(1); }
    if (input->Q == NULL)  { printf("FATAL ERROR: input->Q is NULL (Query non passata?)\n"); exit(1); }
    if (input->P == NULL)  { printf("FATAL ERROR: input->P is NULL (Fit non eseguito?)\n"); exit(1); }
    if (input->index == NULL) { printf("FATAL ERROR: input->index is NULL (Fit non eseguito?)\n"); exit(1); }
    // -----------------------------

    int N = input->N;
    int D = input->D;
    int h = input->h;
    int k = input->k;
    int x = input->x;
    int nq = input->nq;

    printf("DEBUG: Params -> N=%d, D=%d, h=%d, k=%d, x=%d, nq=%d\n", N, D, h, k, x, nq); fflush(stdout);

    // Allocazione Buffer
    int* q_idx = (int*) malloc(x * sizeof(int));
    int* q_sign = (int*) malloc(x * sizeof(int));
    int* p_idx = (int*) malloc(x * sizeof(int));
    int* p_sign = (int*) malloc(x * sizeof(int));
    int* v_idx = (int*) malloc(x * sizeof(int));
    int* v_sign = (int*) malloc(x * sizeof(int));
    
    type* dist_query_pivots = (type*) malloc(h * sizeof(type));
    int* current_knn_ids = (int*) malloc(k * sizeof(int));
    type* current_knn_dists = (type*) malloc(k * sizeof(type));

    printf("DEBUG: Buffer allocati correttamente.\n"); fflush(stdout);

    for (int iq = 0; iq < nq; iq++) {
        // Debug ogni 500 query per non intasare, ma stampa la prima
        if (iq == 0 || iq % 500 == 0) { printf("DEBUG: Processing query %d/%d\n", iq, nq); fflush(stdout); }
        
        type* query_vec = &input->Q[iq * D];

        for (int K = 0; K < k; K++) {
            current_knn_ids[K] = -1;
            current_knn_dists[K] = FLT_MAX;
        }

        // 1. Quantize Query
        quantize_vector(query_vec, D, x, q_idx, q_sign);
        
        // 2. Pivot Distances
        for (int j = 0; j < h; j++) {
            int pivot_row = input->P[j];
            // Controllo bounds pivot
            if (pivot_row < 0 || pivot_row >= N) {
                printf("FATAL: Pivot index %d out of bounds (0..%d)\n", pivot_row, N); exit(1);
            }
            type* pivot_vec = &input->DS[pivot_row * D];
            quantize_vector(pivot_vec, D, x, p_idx, p_sign);
            dist_query_pivots[j] = dist_approx(q_idx, q_sign, p_idx, p_sign, x);
        }

        // 3. Scan Dataset
        for (int v = 0; v < N; v++) {
            type d_k_max = -1.0;
            int max_pos = -1;

            for (int K = 0; K < k; K++) {
                if (current_knn_dists[K] > d_k_max) {
                    d_k_max = current_knn_dists[K];
                    max_pos = K;
                }
            }

            type d_pvt_star = 0.0;
            for (int j = 0; j < h; j++) {
                type d_vp = input->index[v * h + j]; 
                type diff = fabs(d_vp - dist_query_pivots[j]);
                if (diff > d_pvt_star) d_pvt_star = diff;
            }

            if (d_pvt_star < d_k_max) {
                quantize_vector(&input->DS[v * D], D, x, v_idx, v_sign);
                type d_approx_qv = dist_approx(q_idx, q_sign, v_idx, v_sign, x);

                if (d_approx_qv < d_k_max) {
                    current_knn_ids[max_pos] = v;
                    current_knn_dists[max_pos] = d_approx_qv;
                }
            }
        }

        // 4. Refinement
        for (int K = 0; K < k; K++) {
            int id = current_knn_ids[K];
            if (id != -1) {
                type true_dist = euclidean_distance(query_vec, &input->DS[id * D], D);
                input->id_nn[iq * k + K] = id;
                input->dist_nn[iq * k + K] = true_dist;
            } else {
                input->id_nn[iq * k + K] = -1;
                input->dist_nn[iq * k + K] = FLT_MAX;
            }
        }
    }

    printf("DEBUG: Fine predict, inizio free\n"); fflush(stdout);
    free(q_idx); free(q_sign);
    free(p_idx); free(p_sign);
    free(v_idx); free(v_sign);
    free(dist_query_pivots);
    free(current_knn_ids);
    free(current_knn_dists);
}