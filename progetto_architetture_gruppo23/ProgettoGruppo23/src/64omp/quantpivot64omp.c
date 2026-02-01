#include "common.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <xmmintrin.h>

// Misuro approcci diversi
#ifndef QUANTVEC_O
#define QUANTVEC_O -1

#if QUANTVEC_O == -1
extern void quantize_vector(const type *vec, const int D, const int x,
                            int *out_idx, int *out_sign, type *shadow_val);
#else
static void quantize_vector(const type *vec, const int D, const int x,
                            int *out_idx, int *out_sign);
#endif

#endif

#ifndef APPROX_DIST_O
#define APPROX_DIST_O -1

#if APPROX_DIST_O == -1
extern type dist_approx(int *v_idx, int *v_sign, int *w_idx, int *w_sign,
                        int x);
#else
static type dist_approx(int *v_idx, int *v_sign, int *w_idx, int *w_sign,
                        int x);
#endif

#endif

#if QUANTVEC_O == 0
/**
 * Quantizzazione: Trova le X componenti con valore assoluto maggiore.
 * Output: riempie idx_buffer e sign_buffer
 */
static void quantize_vector(const type *vec, const int D, const int x,
                            int *out_idx, int *out_sign) {
  /*
   Approccio:
   - Leggo il vettore una sola volta
   Questa parte probabilmente si può ottimizzare molto
   - Salvo gli 'x' indici con valore assoluto maggiore
   - Salvo il segno in out_sign

   Problema:
   Questa parte forse si potrebbe ottimizzare
   out_sign[i] deve salvare il segno dell'elemento vec[out_idx[i]]
  */
  for (int i = 0; i < x; i++) {
    out_idx[i] = -1;
  }

  for (int i = 0; i < D; i++) {
    type val = vec[i];
    type abs_val = fabs(val);

    // Troviamo la posizione di inserimento
    // manteniamo decrescente per facilità
    int pos = -1;
    for (int j = 0; j < x; j++) {
      if (out_idx[j] == -1 || abs_val > fabs(vec[out_idx[j]])) {
        pos = j;
        break;
      }
    }

    if (pos != -1) {
      // Shift a destra per fare spazio
      for (int k = x - 1; k > pos; k--) {
        out_idx[k] = out_idx[k - 1];
        out_sign[k] = out_sign[k - 1];
      }
      // Inserimento
      out_idx[pos] = i;
      out_sign[pos] = (val >= 0) ? 1 : -1;
    }
  }
}
#endif

#if QUANTVEC_O == 1
/*
 * * Quantize Vector V1
 *
 * Nessuna ottimizzazione sulle strutture dati.
 * Si cerca solo di usare il minor numero possibile di loop
 * e massimizzare il branch predictability.
 * */
static void quantize_vector(const type *vec, const int D, const int x,
                            int *out_idx, int *out_sign) {
  type min_abs_in_buffer = -1.0;
  int min_pos_in_buffer = 0;
  type *shadow_vals = malloc(x * sizeof(type));

  for (int i = 0; i < x; i++) {
    type val = vec[i];
    type abs_val = fabs(val);
    out_idx[i] = i;
    out_sign[i] = (val >= 0) ? 1 : 0;
    shadow_vals[i] = abs_val;
    if (min_abs_in_buffer == -1 || abs_val < min_abs_in_buffer) {
      min_abs_in_buffer = abs_val;
      min_pos_in_buffer = i;
    }
  }

  for (int i = x; i < D; i++) {
    type val = vec[i];
    type abs_val = fabs(val);
    if (abs_val > min_abs_in_buffer) {
      out_idx[min_pos_in_buffer] = i;
      out_sign[min_pos_in_buffer] = (val >= 0) ? 1 : 0;

      shadow_vals[min_pos_in_buffer] = abs_val;

      // Re-find the new minimum in the buffer (O(x))
      // Could be faster using SIMD
      min_abs_in_buffer = (sizeof(type) == sizeof(double)) ? DBL_MAX : FLT_MAX;
      for (int k = 0; k < x; k++) {
        if (shadow_vals[k] < min_abs_in_buffer) {
          min_abs_in_buffer = shadow_vals[k];
          min_pos_in_buffer = k;
        }
      }
    }
  }

  free(shadow_vals);

  for (int i = 1; i < x; i++) {
    int key_idx = out_idx[i];
    int key_sign = out_sign[i];
    int j = i - 1;

    // Sposta gli elementi maggiori della key avanti
    while (j >= 0 && out_idx[j] > key_idx) {
      out_idx[j + 1] = out_idx[j];
      out_sign[j + 1] = out_sign[j]; // Sincronizza il segno!
      j = j - 1;
    }
    out_idx[j + 1] = key_idx;
    out_sign[j + 1] = key_sign;
  }
}
#endif

#if QUANTVEC_O == 2
static inline void swap(int i, int j, type *vals, int *idxs, int *signs) {
  type tmp_val = vals[i];
  vals[i] = vals[j];
  vals[j] = tmp_val;

  int tmp_idx = idxs[i];
  idxs[i] = idxs[j];
  idxs[j] = tmp_idx;

  int tmp_sign = signs[i];
  signs[i] = signs[j];
  signs[j] = tmp_sign;
}

/*
 * Restores the Min-Heap property starting from index `i`.
 * Complexity: O(log X) */
static void sift_down(int i, int n, type *vals, int *idxs, int *signs) {
  while (1) {
    int smallest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && vals[left] < vals[smallest])
      smallest = left;

    if (right < n && vals[right] < vals[smallest])
      smallest = right;

    if (smallest != i) {
      swap(i, smallest, vals, idxs, signs);
      i = smallest; // Continue down the tree
    } else {
      break;
    }
  }
}

/*
 * * Quantize Vector V2
 *
 * Questa funzione rimane generale ma usa una min-heap (implementata con un
 * array) la min heap è gestita tramite due funzioni helper swap e sift_down. La
 * min-heap rende facile trovare e ricordare l'elemento più piccolo trovato. */
static void quantize_vector(const type *vec, const int D, const int x,
                            int *out_idx, int *out_sign) {
  type *shadow_vals = malloc(x * sizeof(type));

  for (int i = 0; i < x; i++) {
    type val = vec[i];
    shadow_vals[i] = fabs(val);
    out_idx[i] = i;
    out_sign[i] = (val >= 0) ? 1 : 0;
  }

  // Build the initial Heap.
  // Start from the last non-leaf node and sift down.
  for (int i = (x / 2) - 1; i >= 0; i--) {
    sift_down(i, x, shadow_vals, out_idx, out_sign);
  }

  for (int i = x; i < D; i++) {
    type val = vec[i];
    type abs_val = fabs(val);

    if (abs_val > shadow_vals[0]) {
      shadow_vals[0] = abs_val;
      out_idx[0] = i;
      out_sign[0] = (val >= 0) ? 1 : 0;

      sift_down(0, x, shadow_vals, out_idx, out_sign);
    }
  }

  free(shadow_vals);

  for (int i = 1; i < x; i++) {
    int key_idx = out_idx[i];
    int key_sign = out_sign[i];
    int j = i - 1;

    // Sposta gli elementi maggiori della key avanti
    while (j >= 0 && out_idx[j] > key_idx) {
      out_idx[j + 1] = out_idx[j];
      out_sign[j + 1] = out_sign[j]; // Sincronizza il segno!
      j = j - 1;
    }
    out_idx[j + 1] = key_idx;
    out_sign[j + 1] = key_sign;
  }
}
#endif

#if QUANTVEC_O == 3
#define ABS_MASK 0x7FFFFFFFFFFFFFFF
/*
 * * Quantize Vector V3
 *
 * Nessuna ottimizzazione sulle strutture dati.
 * Si utilizza la rappresentazione in memoria dei double secondo lo standard
 * IEEE per comparare i valori come se fossero interi e fare operazioni piu
 * rapide
 * */
static void quantize_vector(const type *vec, const int D, const int x,
                            int *out_idx, int *out_sign) {
  int64_t min_abs_in_buffer = -1;
  int min_pos_in_buffer = 0;

  const int64_t *v_int = (const int64_t *)vec;
  int64_t *shadow_vals = malloc(x * sizeof(type));

  for (int i = 0; i < x; i++) {
    int64_t val = v_int[i];
    int64_t abs_val = val & ABS_MASK;
    out_idx[i] = i;
    out_sign[i] = 1 ^ ((uint64_t)val >> 63);
    shadow_vals[i] = abs_val;
    if (min_abs_in_buffer == -1 || abs_val < min_abs_in_buffer) {
      min_abs_in_buffer = abs_val;
      min_pos_in_buffer = i;
    }
  }

  for (int i = x; i < D; i++) {
    int64_t val = v_int[i];
    int64_t abs_val = val & ABS_MASK;
    if (abs_val > min_abs_in_buffer) {
      out_idx[min_pos_in_buffer] = i;
      out_sign[min_pos_in_buffer] = 1 ^ ((uint64_t)val >> 63);

      shadow_vals[min_pos_in_buffer] = abs_val;

      // Re-find the new minimum in the buffer (O(x))
      // Could be faster using SIMD
      min_abs_in_buffer = shadow_vals[0];
      min_pos_in_buffer = 0;
      for (int k = 1; k < x; k++) {
        int64_t curr = shadow_vals[k];
        if (curr < min_abs_in_buffer) {
          min_abs_in_buffer = curr;
          min_pos_in_buffer = k;
        }
      }
    }
  }

  free(shadow_vals);

  for (int i = 1; i < x; i++) {
    int key_idx = out_idx[i];
    int key_sign = out_sign[i];
    int j = i - 1;

    // Sposta gli elementi maggiori della key avanti
    while (j >= 0 && out_idx[j] > key_idx) {
      out_idx[j + 1] = out_idx[j];
      out_sign[j + 1] = out_sign[j]; // Sincronizza il segno!
      j = j - 1;
    }
    out_idx[j + 1] = key_idx;
    out_sign[j + 1] = key_sign;
  }
}
#endif

#if APPROX_DIST_O == 0
/**
 * Calcolo esplicito dei 4 termini della distanza approssimata.
 * @param v_idx   Indici degli elementi non-zero di v
 * @param v_sign  Segni di v (1 = positivo/v+, 0 = negativo/v-)
 * @param w_idx   Indici degli elementi non-zero di w
 * @param w_sign  Segni di w (1 = positivo/w+, 0 = negativo/w-)
 * @param x       Numero di elementi salvati (quantizzazione)
 */
static type dist_approx(int *v_idx, int *v_sign, int *w_idx, int *w_sign,
                        int x) {
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
        int is_v_plus = (v_sign[i] == 1);
        int is_v_minus = (v_sign[i] == 0); // o else

        // Determiniamo a quale parte del vettore appartiene w (w+ o w-)
        int is_w_plus = (w_sign[j] == 1);
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
#endif

#if APPROX_DIST_O == 1
static type dist_approx(int *v_idx, int *v_sign, int *w_idx, int *w_sign,
                        int x) {
  type result = 0.0;
  int i = 0;
  int j = 0;

  while (i < x && j < x) {
    int vi = v_idx[i];
    int wj = w_idx[j];

    if (vi == wj) {
      int sign_diff = v_sign[i] ^ w_sign[j];
      result += (1.0 - 2.0 * sign_diff);
      i++;
      j++;
    } else {
      int vi_smaller = (vi < wj);
      i += vi_smaller;
      j += !vi_smaller;
    }
  }
  return result;
}
#endif

// Necessaria per il passaggio finale della funzione predict
static type euclidean_distance(type *v, type *w, int D) {
  type sum = 0.0;
  for (int i = 0; i < D; i++) {
    type diff = v[i] - w[i];
    sum += diff * diff;
  }
  return sqrt(sum);
}

void fit(params *input) {
  const int N = input->N;
  const int D = input->D;
  const int h = input->h;
  const int x = input->x;

  input->P = (int *)malloc(h * sizeof(int));
  input->index = (type *)_mm_malloc(N * h * sizeof(type), align);

  const int step = N / h;
  for (int j = 0; j < h; j++) {
    input->P[j] = step * j;
  }

  int *pivot_idxs = (int *)_mm_malloc(h * x * sizeof(int), align);
  int *pivot_signs = (int *)_mm_malloc(h * x * sizeof(int), align);

  // Pre-quantizzazione dei pivot.
  // #pragma omp parallel
  {
#if QUANTVEC_O == -1
    type *th_shadow_vals = (double *)_mm_malloc(x * sizeof(double), align);
#endif
    // #pragma omp for
    for (int j = 0; j < h; j++) {
      type *p_vec = &input->DS[input->P[j] * D];
#if QUANTVEC_O == -1
      quantize_vector(p_vec, D, x, &pivot_idxs[j * x], &pivot_signs[j * x],
                      th_shadow_vals);
#else
      quantize_vector(p_vec, D, x, &pivot_idxs[j * x], &pivot_signs[j * x]);
#endif
    }
  }

  // Le variabili dichiarate qui sono private per ogni thread
  // #pragma omp parallel
  {
    int *th_point_idx = (int *)_mm_malloc(x * sizeof(int), align);
    int *th_point_sign = (int *)_mm_malloc(x * sizeof(int), align);

#if QUANTVEC_O == -1
    type *th_shadow_vals = (double *)_mm_malloc(x * sizeof(double), align);
#endif

    // #pragma omp for
    for (int i = 0; i < N; i++) {

#if QUANTVEC_O == -1
      quantize_vector(&input->DS[i * D], D, x, th_point_idx, th_point_sign,
                      th_shadow_vals);
#else
      quantize_vector(&input->DS[i * D], D, x, th_point_idx, th_point_sign);
#endif

      type *out_row = &input->index[i * h];

      for (int j = 0; j < h; j++) {
        type dist = dist_approx(
            th_point_idx, th_point_sign,             // Vettore Thread-Local
            &pivot_idxs[j * x], &pivot_signs[j * x], // Vettore Pivot
            x);

        out_row[j] = dist;
      }
    }

    _mm_free(th_point_idx);
    _mm_free(th_point_sign);
  }

  _mm_free(pivot_idxs);
  _mm_free(pivot_signs);
}

void predict(params *input) {
  const int N = input->N;
  const int D = input->D;
  const int h = input->h;
  const int k = input->k;
  const int x = input->x;
  const int nq = input->nq;

  // ---------------------------------------------------
  // 1. PRE-QUANTIZZAZIONE PIVOT (Calcolati una sola volta!)
  // ---------------------------------------------------
  // Invece di ricalcolarli per ogni query, li prepariamo qui.
  int *pivot_idxs = (int *)_mm_malloc(h * x * sizeof(int), align);
  int *pivot_signs = (int *)_mm_malloc(h * x * sizeof(int), align);

  // Buffer temporaneo per la quantizzazione dei pivot (usato solo qui)
  // Nota: qui serve shadow_vals se usiamo la versione AVX
  // #pragma omp parallel
  {
#if QUANTVEC_O == -1
    double *tmp_shadow = (double *)_mm_malloc(x * sizeof(double), align);
#endif

    // Parallelizziamo la pre-elaborazione dei pivot
    // #pragma omp for
    for (int j = 0; j < h; j++) {
      type *p_vec = &input->DS[input->P[j] * D];

#if QUANTVEC_O == -1
      quantize_vector(p_vec, D, x, &pivot_idxs[j * x], &pivot_signs[j * x],
                      tmp_shadow);
#else
      quantize_vector(p_vec, D, x, &pivot_idxs[j * x], &pivot_signs[j * x]);
#endif
    }

#if QUANTVEC_O == -1
    _mm_free(tmp_shadow);
#endif
  }

  // ---------------------------------------------------
  // 2. LOOP PARALLELO SULLE QUERY
  // ---------------------------------------------------
  // #pragma omp parallel
  {
    // --- Allocazione Thread-Local (Una tantum per thread) ---
    // Vettori quantizzati: Query (q) e Candidato Dataset (v)
    int *th_q_idx = (int *)_mm_malloc(x * sizeof(int), align);
    int *th_q_sign = (int *)_mm_malloc(x * sizeof(int), align);
    int *th_v_idx = (int *)_mm_malloc(x * sizeof(int), align);
    int *th_v_sign = (int *)_mm_malloc(x * sizeof(int), align);

    // Distanze Query-Pivots
    type *th_dist_query_pivots = (type *)_mm_malloc(h * sizeof(type), align);

    // Lista K-NN locale
    int *th_knn_ids = (int *)malloc(k * sizeof(int));
    type *th_knn_dists = (type *)malloc(k * sizeof(type));

// Buffer shadow per quantize_vector ottimizzato
#if QUANTVEC_O == -1
    double *th_shadow_vals = (double *)_mm_malloc(x * sizeof(double), align);
#endif

    // Ciclo su tutte le query assegnate a questo thread
    // #pragma omp for
    for (int iq = 0; iq < nq; iq++) {
      type *query_vec = &input->Q[iq * D];

      // A. Reset Lista K-NN
      for (int K = 0; K < k; K++) {
        th_knn_ids[K] = -1;
        th_knn_dists[K] = FLT_MAX;
      }
      type d_k_max = FLT_MAX; // Distanza massima corrente nella lista

// B. Quantizzazione Query
#if QUANTVEC_O == -1
      quantize_vector(query_vec, D, x, th_q_idx, th_q_sign, th_shadow_vals);
#else
      quantize_vector(query_vec, D, x, th_q_idx, th_q_sign);
#endif

      // C. Calcola distanze Query <-> Tutti i Pivot
      for (int j = 0; j < h; j++) {
        // Usiamo i pivot pre-calcolati (globali read-only)
        th_dist_query_pivots[j] = dist_approx(
            th_q_idx, th_q_sign, &pivot_idxs[j * x], &pivot_signs[j * x], x);
      }

      // D. Scansione Dataset (Candidate Filtering)
      for (int v = 0; v < N; v++) {

        // 1. Lower Bound (Triangular Inequality)
        // d_pvt* = max |d_approx(v, p) - d_approx(q, p)|
        type d_pvt_star = 0.0;

        // Nota: input->index è accessibile in lettura da tutti i thread senza
        // lock
        type *v_dists_ptr = &input->index[v * h];

        // Loop unrolling manuale (opzionale, il compilatore spesso lo fa)
        for (int j = 0; j < h; j++) {
          type diff = fabs(v_dists_ptr[j] - th_dist_query_pivots[j]);
          if (diff > d_pvt_star) {
            d_pvt_star = diff;
            // Ottimizzazione "Early Exit" sul filtro?
            // Se d_pvt_star supera già d_k_max, possiamo uscire dal loop j?
            // Sì, perché il max può solo crescere.
            if (d_pvt_star >= d_k_max)
              break;
          }
        }

        // 2. Primo Filtro
        if (d_pvt_star < d_k_max) {

// Quantizza il candidato v (riutilizzando i buffer thread-local)
#if QUANTVEC_O == -1
          quantize_vector(&input->DS[v * D], D, x, th_v_idx, th_v_sign,
                          th_shadow_vals);
#else
          quantize_vector(&input->DS[v * D], D, x, th_v_idx, th_v_sign);
#endif

          // Calcola distanza approssimata reale
          type d_approx_qv =
              dist_approx(th_q_idx, th_q_sign, th_v_idx, th_v_sign, x);

          // 3. Secondo Filtro e Aggiornamento K-NN
          if (d_approx_qv < d_k_max) {
            // Trova l'indice del peggiore attuale (max_pos)
            // Poiché k è piccolo, scansione lineare va bene
            int max_pos = 0;
            type current_max = th_knn_dists[0];
            for (int K = 1; K < k; K++) {
              if (th_knn_dists[K] > current_max) {
                current_max = th_knn_dists[K];
                max_pos = K;
              }
            }

            // Sostituisci
            th_knn_ids[max_pos] = v;
            th_knn_dists[max_pos] = d_approx_qv;

            // Aggiorna d_k_max per i prossimi check
            // Dobbiamo ritrovare il nuovo massimo
            d_k_max = -1.0;
            for (int K = 0; K < k; K++) {
              if (th_knn_dists[K] > d_k_max)
                d_k_max = th_knn_dists[K];
            }
          }
        }
      } // Fine loop Dataset (v)

      // E. Raffinamento Finale (Distanza Euclidea Reale)
      for (int K = 0; K < k; K++) {
        int id = th_knn_ids[K];
        if (id != -1) {
          type true_dist = euclidean_distance(query_vec, &input->DS[id * D], D);
          input->id_nn[iq * k + K] = id;
          input->dist_nn[iq * k + K] = true_dist;
        } else {
          input->id_nn[iq * k + K] = -1;
          input->dist_nn[iq * k + K] = FLT_MAX;
        }
      }

      // Opzionale: Ordinare i risultati finali per distanza (spesso richiesto
      // nei KNN) Non incluso per mantenere fedeltà alla tua logica originale.

    } // Fine loop Queries (iq)

    // --- Pulizia Thread-Local ---
    _mm_free(th_q_idx);
    _mm_free(th_q_sign);
    _mm_free(th_v_idx);
    _mm_free(th_v_sign);
    _mm_free(th_dist_query_pivots);
    free(th_knn_ids);
    free(th_knn_dists);
#if QUANTVEC_O == -1
    _mm_free(th_shadow_vals);
#endif

  } // Fine Regione Parallela

  // Pulizia Globale
  _mm_free(pivot_idxs);
  _mm_free(pivot_signs);
}
