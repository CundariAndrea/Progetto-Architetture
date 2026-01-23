#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <xmmintrin.h>
#include <string.h> // per memset
#include "common.h"
#include <time.h>


static inline double now_sec(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec *1e-9;

}


/**
 * Quantizzazione: Trova le X componenti con valore assoluto maggiore.
 * Output: riempie idx_buffer e sign_buffer
 */
static void quantize_vector(const type* vec, int D, int x, int* out_idx, int* out_sign) {
    // Manteniamo una lista top-x di (abs, idx, sign) già materializzata
    // Così evitiamo fabs(vec[out_idx[j]]) dentro il loop interno.
    type* best_abs = (type*)alloca((size_t)x * sizeof(type));

    for (int i = 0; i < x; i++) {
        out_idx[i]  = -1;
        out_sign[i] = 1;
        best_abs[i] = (type)-1.0;   // sentinella
    }

    for (int i = 0; i < D; i++) {
        type val = vec[i];
        type abs_val = (type)fabs(val);

        // trova posizione di inserimento nella top-x (ordine decrescente)
        int pos = -1;
        for (int j = 0; j < x; j++) {
            if (abs_val > best_abs[j]) { pos = j; break; }
        }
        if (pos < 0) continue;

        // shift verso destra
        for (int k = x - 1; k > pos; k--) {
            best_abs[k] = best_abs[k - 1];
            out_idx[k]  = out_idx[k - 1];
            out_sign[k] = out_sign[k - 1];
        }

        best_abs[pos] = abs_val;
        out_idx[pos]  = i;
        out_sign[pos] = (val >= 0) ? 1 : -1;
    }
}


/**
 * Calcolo esplicito dei 4 termini della distanza approssimata.
 * @param v_idx   Indici degli elementi non-zero di v
 * @param v_sign  Segni di v (1 = positivo/v+, 0 = negativo/v-)
 * @param w_idx   Indici degli elementi non-zero di w
 * @param w_sign  Segni di w (1 = positivo/w+, 0 = negativo/w-)
 * @param x       Numero di elementi salvati (quantizzazione)
 */
type dist_approx(int* v_idx, int* v_sign, int* w_idx, int* w_sign, int x) {
    /*
    Questa funzione è volutamente non ottimizzata e rispecchia il pdf
    Si può ottimizzare molto e va fatto in nasm
    Inutile complicarla qui
    Esempio possibile ottimizzazione:
    - Calcolo prodotto scalare tramite |a|*|b|*cos(phi)
    */
    // 1. Inizializziamo i 4 accumulatori dell'Eq. (2)
    type dot_vp_wp = 0.0; // (v+ * w+)
    type dot_vm_wm = 0.0; // (v- * w-)
    type dot_vp_wm = 0.0; // (v+ * w-)
    type dot_vm_wp = 0.0; // (v- * w+)

    // 2. Doppio ciclo per trovare le intersezioni 
    // prodotto scalare completo
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < x; j++) {
            // Il prodotto è non-zero solo se gli indici coincidono
            // v = [0 1 1] w = [0 0 1] v*w = 0*0 + 0*1 + 1*1
            if (v_idx[i] == w_idx[j]) {
                // Determiniamo a quale parte del vettore appartiene v (v+ o v-)
                int is_v_plus  = (v_sign[i] == 1);
                int is_v_minus = (v_sign[i] == 0); // o else

                // Determiniamo a quale parte del vettore appartiene w (w+ o w-)
                int is_w_plus  = (w_sign[j] == 1);
                int is_w_minus = (w_sign[j] == 0); // o else

                // 3. Aggiorniamo il termine corrispondente
                // Caso: v+ e w+
                if (is_v_plus && is_w_plus) {
                    dot_vp_wp += 1.0;
                }
                // Caso: v- e w-
                else if (is_v_minus && is_w_minus) {
                    dot_vm_wm += 1.0;
                }
                // Caso: v+ e w-
                else if (is_v_plus && is_w_minus) {
                    dot_vp_wm += 1.0;
                }
                // Caso: v- e w+
                else if (is_v_minus && is_w_plus) {
                    dot_vm_wp += 1.0;
                }
            }
        }
    }

    // 4. Applichiamo la formula finale (Eq. 2 del PDF)
    // d = (v+ * w+) + (v- * w-) - (v+ * w-) - (v- * w+)
    type result = dot_vp_wp + dot_vm_wm - dot_vp_wm - dot_vm_wp;
    return result;
}

// Necessaria per il passaggio finale della funzione predict
static type euclidean_distance(type* v, type* w, int D) {
    type sum = 0.0;
    for (int i = 0; i < D; i++) {
        type diff = v[i] - w[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

//  FIT (Indexing)
void fit(params* input) {
    int N = input->N;
    int D = input->D;
    int h = input->h;
    int x = input->x;

    // 1. Allocazione Pivot P
    // Salviamo solo gli indici delle righe in cui input->DS[.] ha un pivot
    input->P = (int*) malloc(h * sizeof(int));
    // Selezione Pivot: i = floor(N/h) * j
    int step = N / h;
    for (int j = 0; j < h; j++) {
        input->P[j] = step * j;
    }

    // 2. Allocazione Matrice Index (N x h)
    // Memorizza la distanza approssimata tra ogni punto e ogni pivot
    input->index = (MATRIX) _mm_malloc(N * h * sizeof(type), align);

    // 3. Buffer Temporanei per Quantizzazione
    // Si pre-alloca tutta la memoria per aumentare l'efficienza delle funzioni ausiliarie
    int* pivot_idxs = malloc(h * x * sizeof(int));
    int* pivot_signs = malloc(h * x * sizeof(int));
    int* point_idx = malloc(x * sizeof(int));
    int* point_sign = malloc(x * sizeof(int));

    // Pre-quantizzazione di tutti i pivot
    for (int j = 0; j < h; j++) {
        // Moltiplico per D perché la matrice è salvata in forma di array
        type* p_vec = &input->DS[input->P[j] * D];
        quantize_vector(p_vec, D, x, &pivot_idxs[j*x], &pivot_signs[j*x]);
    }

    // 4. Costruzione Indice
    for (int i = 0; i < N; i++) {
        // Quantizza punto corrente
        quantize_vector(&input->DS[i * D], D, x, point_idx, point_sign);

        // Calcola distanze con tutti i pivot
        for (int j = 0; j < h; j++) {
            // Prima il punto di DS e poi il pivot (Da formula)
            type dist = dist_approx(point_idx, point_sign, &pivot_idxs[j*x], &pivot_signs[j*x], x);
            input->index[i * h + j] = dist;
        }
    }

    free(pivot_idxs);
    free(pivot_signs);
    free(point_idx);
    free(point_sign);
}

//  PREDICT (Querying)
void predict(params* input) {
    int N  = input->N;
    int D  = input->D;
    int h  = input->h;
    int k  = input->k;
    int x  = input->x;
    int nq = input->nq;

    // Timer accumulati (Step A)
    double t_quant      = 0.0;
    double t_distapprox = 0.0;
    double t_lowerbound = 0.0;
    double t_euclid     = 0.0;

    // 1. Allocazione buffer temporanei
    int* q_idx  = (int*) malloc(x * sizeof(int));
    int* q_sign = (int*) malloc(x * sizeof(int));
    int* p_idx  = (int*) malloc(x * sizeof(int));
    int* p_sign = (int*) malloc(x * sizeof(int));
    int* v_idx  = (int*) malloc(x * sizeof(int));
    int* v_sign = (int*) malloc(x * sizeof(int));

    type* dist_query_pivots = (type*) malloc(h * sizeof(type));

    int*  current_knn_ids   = (int*)  malloc(k * sizeof(int));
    type* current_knn_dists = (type*) malloc(k * sizeof(type));

    //test ottimizzazione #1
    int* pivot_idxs  = (int*) malloc(h * x * sizeof(int));
    int* pivot_signs = (int*) malloc(h * x * sizeof(int));

    for (int j = 0; j < h; j++) {
        int pivot_row = input->P[j];
        type* pivot_vec = &input->DS[pivot_row * D];

        double t0 = now_sec();
        quantize_vector(pivot_vec, D, x, &pivot_idxs[j * x], &pivot_signs[j * x]);
        t_quant += now_sec() - t0;
    }

    // fine test #1

    // Foreach Query
    for (int iq = 0; iq < nq; iq++) {
        type* query_vec = &input->Q[iq * D];

        // 1. Inizializza la lista K-NN a {-1, +infinity}
        for (int K = 0; K < k; K++) {
            current_knn_ids[K]   = -1;
            current_knn_dists[K] = FLT_MAX;
        }

        // quantize query
        {
            double t0 = now_sec();
            quantize_vector(query_vec, D, x, q_idx, q_sign);
            t_quant += now_sec() - t0;
        }

        // pre-quantizzazione pivot (una volta sola)
        for (int j = 0; j < h; j++) {
            double t2 = now_sec();
            dist_query_pivots[j] = dist_approx(q_idx, q_sign,
                                            &pivot_idxs[j * x], &pivot_signs[j * x],
                                            x);
            t_distapprox += now_sec() - t2;
        }

        // 3. Scansiona tutto il dataset per trovare i candidati
        for (int v = 0; v < N; v++) {

            // Trova la distanza massima attuale nella lista K-NN (d_k^max)
            type d_k_max = -1.0;
            int max_pos = -1;

            for (int K = 0; K < k; K++) {
                if (current_knn_dists[K] > d_k_max) {
                    d_k_max = current_knn_dists[K];
                    max_pos = K;
                }
            }

            // 4. Lower bound (loop su h)
            type d_pvt_star;
            {
                double t3 = now_sec();
                d_pvt_star = 0.0;
                for (int j = 0; j < h; j++) {
                    type d_vp = input->index[v * h + j];
                    type diff = fabs(d_vp - dist_query_pivots[j]);
                    if (diff > d_pvt_star) d_pvt_star = diff;
                }
                t_lowerbound += now_sec() - t3;
            }

            // 5. Primo filtro
            if (d_pvt_star < d_k_max) {

                // quantize v
                {
                    double t4 = now_sec();
                    quantize_vector(&input->DS[v * D], D, x, v_idx, v_sign);
                    t_quant += now_sec() - t4;
                }

                // dist approx query-v
                type d_approx_qv;
                {
                    double t5 = now_sec();
                    d_approx_qv = dist_approx(q_idx, q_sign, v_idx, v_sign, x);
                    t_distapprox += now_sec() - t5;
                }

                // 6. Secondo filtro + update
                if (d_approx_qv < d_k_max) {
                    current_knn_ids[max_pos]   = v;
                    current_knn_dists[max_pos] = d_approx_qv;
                }
            }
        }

        // 7. Raffinamento: distanze Euclidee REALI per i K candidati
        for (int K = 0; K < k; K++) {
            int id = current_knn_ids[K];
            if (id != -1) {
                type true_dist;
                {
                    double t6 = now_sec();
                    true_dist = euclidean_distance(query_vec, &input->DS[id * D], D);
                    t_euclid += now_sec() - t6;
                }

                input->id_nn[iq * k + K]   = id;
                input->dist_nn[iq * k + K] = true_dist;
            } else {
                input->id_nn[iq * k + K]   = -1;
                input->dist_nn[iq * k + K] = FLT_MAX;
            }
        }
    }

    // stampa una sola volta (Step A)
    printf("\n[TIMERS] quantize_vector:    %.6f s\n", t_quant);
    printf("[TIMERS] dist_approx:        %.6f s\n", t_distapprox);
    printf("[TIMERS] lower_bound loop:   %.6f s\n", t_lowerbound);
    printf("[TIMERS] euclidean_distance: %.6f s\n\n", t_euclid);

    free(pivot_idxs);
    free(pivot_signs);
    free(q_idx); free(q_sign);
    free(p_idx); free(p_sign);
    free(v_idx); free(v_sign);
    free(dist_query_pivots);
    free(current_knn_ids);
    free(current_knn_dists);
}
