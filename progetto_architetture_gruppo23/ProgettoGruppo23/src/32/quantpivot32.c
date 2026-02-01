#define _POSIX_C_SOURCE 200809L
#include "common.h"
#include <alloca.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h> // per memset
#include <time.h>
#include <xmmintrin.h>

static __thread double g_t_abs_sse = 0.0; // calcolo il tempo di abs_sse32

extern void abs_sse32(float *out_abs, const float *in, int D);

static inline void abs_sse32_safe(float *out_abs, const float *in, int D) {
  int D4 = D & ~3;
  if (D4 > 0)
    abs_sse32(out_abs, in, D4); // chiamata a funzione esterna OK
  for (int i = D4; i < D; i++) {
    float v = in[i];
    out_abs[i] = (v < 0.0f) ? -v : v;
  }
}

static inline double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/**
 * Quantizzazione: Trova le X componenti con valore assoluto maggiore.
 * Output: riempie idx_buffer e sign_buffer
 */

static void quantize_vector(const type *vec, int D, int x, int *out_idx,
                            int *out_sign) {
  // buffer per valori assoluti calcolati in SSE
  float *absvals = (float *)alloca((size_t)D * sizeof(float));
  double t0 = now_sec();
  abs_sse32_safe(absvals, vec, D);
  g_t_abs_sse += now_sec() - t0;
  // top-x
  type *best_abs = (type *)alloca((size_t)x * sizeof(type));

  for (int i = 0; i < x; i++) {
    out_idx[i] = -1;
    out_sign[i] = 1;
    best_abs[i] = (type)-1.0f;
  }

  for (int i = 0; i < D; i++) {
    type abs_val = (type)absvals[i]; // <-- usa l'output SSE
    type val = vec[i];

    int pos = -1;
    for (int j = 0; j < x; j++) {
      if (abs_val > best_abs[j]) {
        pos = j;
        break;
      }
    }
    if (pos < 0)
      continue;

    for (int k = x - 1; k > pos; k--) {
      best_abs[k] = best_abs[k - 1];
      out_idx[k] = out_idx[k - 1];
      out_sign[k] = out_sign[k - 1];
    }

    best_abs[pos] = abs_val;
    out_idx[pos] = i;
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

#include <stdint.h>
#include <stdlib.h>

static inline type dist_approx_fast(const int *v_idx, const int *v_sign,
                                    const int *w_idx, const int *w_sign, int x,
                                    int D) {
  // Buffer per-thread, ridimensionabili
  static __thread uint32_t *seen = NULL;
  static __thread int8_t *vsgn = NULL;
  static __thread uint32_t cap = 0;
  static __thread uint32_t stamp = 1;

  // (Ri)alloc se D cresce
  if ((uint32_t)D > cap) {
    uint32_t newcap = (cap == 0) ? 256 : cap;
    while (newcap < (uint32_t)D)
      newcap *= 2;

    seen = (uint32_t *)realloc(seen, newcap * sizeof(uint32_t));
    vsgn = (int8_t *)realloc(vsgn, newcap * sizeof(int8_t));

    // inizializza seen a 0 per tutta la nuova capacità
    // (solo quando riallochiamo)
    for (uint32_t i = cap; i < newcap; i++)
      seen[i] = 0;

    cap = newcap;
  }

  // gestione overflow stamp (raro ma corretto)
  stamp++;
  if (stamp == 0) {
    // azzera seen e riparti
    for (uint32_t i = 0; i < cap; i++)
      seen[i] = 0;
    stamp = 1;
  }

  // carica v in O(x)
  for (int i = 0; i < x; i++) {
    int idx = v_idx[i];
    if ((unsigned)idx < (unsigned)D) {
      seen[idx] = stamp;
      vsgn[idx] = (int8_t)v_sign[i]; // attesi: +1 o -1
    }
  }

  int dot_vp_wp = 0, dot_vm_wm = 0, dot_vp_wm = 0, dot_vm_wp = 0;

  // scansiona w in O(x)
  for (int j = 0; j < x; j++) {
    int idx = w_idx[j];
    if ((unsigned)idx < (unsigned)D && seen[idx] == stamp) {
      int vs = vsgn[idx];
      int ws = w_sign[j];

      // versione coerente con la tua logica: plus se +1, minus altrimenti
      int is_v_plus = (vs == 1);
      int is_v_minus = !is_v_plus;
      int is_w_plus = (ws == 1);
      int is_w_minus = !is_w_plus;

      if (is_v_plus && is_w_plus)
        dot_vp_wp++;
      else if (is_v_minus && is_w_minus)
        dot_vm_wm++;
      else if (is_v_plus && is_w_minus)
        dot_vp_wm++;
      else if (is_v_minus && is_w_plus)
        dot_vm_wp++;
    }
  }

  return (type)(dot_vp_wp + dot_vm_wm - dot_vp_wm - dot_vm_wp);
}

// Necessaria per il passaggio finale della funzione predict
static type euclidean_distance(type *v, type *w, int D) {
  type sum = 0.0;
  for (int i = 0; i < D; i++) {
    type diff = v[i] - w[i];
    sum += diff * diff;
  }
  return sqrt(sum);
}

//  FIT (Indexing)
void fit(params *input) {
  int N = input->N;
  int D = input->D;
  int h = input->h;
  int x = input->x;

  // 1. Allocazione Pivot P
  // Salviamo solo gli indici delle righe in cui input->DS[.] ha un pivot
  input->P = (int *)malloc(h * sizeof(int));
  // Selezione Pivot: i = floor(N/h) * j
  int step = N / h;
  for (int j = 0; j < h; j++) {
    input->P[j] = step * j;
  }

  // 2. Allocazione Matrice Index (N x h)
  // Memorizza la distanza approssimata tra ogni punto e ogni pivot
  input->index = (MATRIX)_mm_malloc(N * h * sizeof(type), align);

  // 3. Buffer Temporanei per Quantizzazione
  // Si pre-alloca tutta la memoria per aumentare l'efficienza delle funzioni
  // ausiliarie
  int *pivot_idxs = malloc(h * x * sizeof(int));
  int *pivot_signs = malloc(h * x * sizeof(int));
  int *point_idx = malloc(x * sizeof(int));
  int *point_sign = malloc(x * sizeof(int));

  // Pre-quantizzazione di tutti i pivot
  for (int j = 0; j < h; j++) {
    // Moltiplico per D perché la matrice è salvata in forma di array
    type *p_vec = &input->DS[input->P[j] * D];
    quantize_vector(p_vec, D, x, &pivot_idxs[j * x], &pivot_signs[j * x]);
  }

  // 4. Costruzione Indice
  for (int i = 0; i < N; i++) {
    // Quantizza punto corrente
    quantize_vector(&input->DS[i * D], D, x, point_idx, point_sign);

    // Calcola distanze con tutti i pivot
    for (int j = 0; j < h; j++) {
      // Prima il punto di DS e poi il pivot (Da formula)
      type dist = dist_approx_fast(point_idx, point_sign, &pivot_idxs[j * x],
                                   &pivot_signs[j * x], x, D);
      input->index[i * h + j] = dist;
    }
  }

  free(pivot_idxs);
  free(pivot_signs);
  free(point_idx);
  free(point_sign);
}

//  PREDICT (Querying)
void predict(params *input) {
  int N = input->N;
  int D = input->D;
  int h = input->h;
  int k = input->k;
  int x = input->x;
  int nq = input->nq;

  // Timer accumulati (Step A)
  double t_quant = 0.0;
  double t_distapprox = 0.0;
  double t_lowerbound = 0.0;
  double t_euclid = 0.0;

  // 1. Allocazione buffer temporanei
  int *q_idx = (int *)malloc(x * sizeof(int));
  int *q_sign = (int *)malloc(x * sizeof(int));
  int *p_idx = (int *)malloc(x * sizeof(int));
  int *p_sign = (int *)malloc(x * sizeof(int));
  int *v_idx = (int *)malloc(x * sizeof(int));
  int *v_sign = (int *)malloc(x * sizeof(int));

  type *dist_query_pivots = (type *)malloc(h * sizeof(type));

  int *current_knn_ids = (int *)malloc(k * sizeof(int));
  type *current_knn_dists = (type *)malloc(k * sizeof(type));

  // test ottimizzazione #1
  int *pivot_idxs = (int *)malloc(h * x * sizeof(int));
  int *pivot_signs = (int *)malloc(h * x * sizeof(int));

  for (int j = 0; j < h; j++) {
    int pivot_row = input->P[j];
    type *pivot_vec = &input->DS[pivot_row * D];

    double t0 = now_sec();
    quantize_vector(pivot_vec, D, x, &pivot_idxs[j * x], &pivot_signs[j * x]);
    t_quant += now_sec() - t0;
  }

  // fine test #1

  // Foreach Query
  for (int iq = 0; iq < nq; iq++) {
    type *query_vec = &input->Q[iq * D];

    // 1. Inizializza la lista K-NN a {-1, +infinity}
    for (int K = 0; K < k; K++) {
      current_knn_ids[K] = -1;
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
      dist_query_pivots[j] = dist_approx_fast(q_idx, q_sign, &pivot_idxs[j * x],
                                              &pivot_signs[j * x], x, D);
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
          if (diff > d_pvt_star)
            d_pvt_star = diff;
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
          d_approx_qv = dist_approx_fast(q_idx, q_sign, v_idx, v_sign, x, D);
          t_distapprox += now_sec() - t5;
        }

        // 6. Secondo filtro + update
        if (d_approx_qv < d_k_max) {
          current_knn_ids[max_pos] = v;
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

        input->id_nn[iq * k + K] = id;
        input->dist_nn[iq * k + K] = true_dist;
      } else {
        input->id_nn[iq * k + K] = -1;
        input->dist_nn[iq * k + K] = FLT_MAX;
      }
    }
  }

  // stampa una sola volta (Step A)
  g_t_abs_sse = 0.0;

  free(pivot_idxs);
  free(pivot_signs);
  free(q_idx);
  free(q_sign);
  free(p_idx);
  free(p_sign);
  free(v_idx);
  free(v_sign);
  free(dist_query_pivots);
  free(current_knn_ids);
  free(current_knn_dists);
}
