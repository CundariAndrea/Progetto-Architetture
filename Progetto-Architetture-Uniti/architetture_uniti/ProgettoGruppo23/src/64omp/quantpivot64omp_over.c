#include "common.h"
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- STATIC CACHE FOR PRE-QUANTIZED DATASET ---
// Since we likely cannot modify 'params' in common.h, we use static globals
// to store the quantized representation of the dataset created in fit().
static int *g_ds_idx = NULL;
static int *g_ds_sign = NULL;

// Comparison function for qsort to sort indices for O(x) intersection
int compare_ints(const void *a, const void *b) {
  return (*(int *)a - *(int *)b);
}

/**
 * Quantization: Finds the X components with largest absolute values.
 * OPTIMIZATION:
 * 1. Returns indices sorted in ascending order (crucial for linear
 * intersection).
 * 2. Optimized selection logic.
 */
static void quantize_vector(const type *vec, int D, int x, int *out_idx,
                            int *out_sign) {
  // Temp structure to track top x values.
  // Since x is usually small (e.g., 16-64), a simple unsorted array +
  // min-search is fast enough and vectorizes better than a heap for very small
  // N.

  // Initialize with -1
  for (int i = 0; i < x; i++) {
    out_idx[i] = -1;
  }

  // Pass 1: Find Top X largest absolute values
  // We maintain a "min-threshold" of the current top X to avoid checking
  // everyone
  type min_abs_in_buffer = -1.0;
  int min_pos_in_buffer = 0;

  for (int i = 0; i < D; i++) {
    type val = vec[i];
    type abs_val = fabs(val);

    // Optimization: Quick check against current minimum in our top-x buffer
    if (abs_val > min_abs_in_buffer) {
      // If buffer isn't full yet (initial phase)
      if (out_idx[x - 1] == -1) {
        // Fill empty slots linearly
        for (int k = 0; k < x; k++) {
          if (out_idx[k] == -1) {
            out_idx[k] = i;
            out_sign[k] =
                (val >= 0) ? 1 : 0; // Store 0 for negative, 1 for positive
            break;
          }
        }
        // If we just filled the last slot, calc the min
        if (out_idx[x - 1] != -1) {
          min_abs_in_buffer = FLT_MAX;
          for (int k = 0; k < x; k++) {
            type a = fabs(vec[out_idx[k]]);
            if (a < min_abs_in_buffer) {
              min_abs_in_buffer = a;
              min_pos_in_buffer = k;
            }
          }
        }
      } else {
        // Buffer is full, replace the minimum element
        out_idx[min_pos_in_buffer] = i;
        out_sign[min_pos_in_buffer] = (val >= 0) ? 1 : 0;

        // Re-find the new minimum in the buffer (O(x))
        min_abs_in_buffer = FLT_MAX;
        for (int k = 0; k < x; k++) {
          type a = fabs(vec[out_idx[k]]);
          if (a < min_abs_in_buffer) {
            min_abs_in_buffer = a;
            min_pos_in_buffer = k;
          }
        }
      }
    }
  }

  // Pass 2: Sort by Index.
  // We need to keep idx and sign paired.
  // Since x is tiny, Bubble Sort is actually faster than qsort overhead
  // usually, but let's write a simple sort.
  for (int i = 0; i < x - 1; i++) {
    for (int j = 0; j < x - i - 1; j++) {
      if (out_idx[j] > out_idx[j + 1]) {
        // Swap Index
        int ti = out_idx[j];
        out_idx[j] = out_idx[j + 1];
        out_idx[j + 1] = ti;
        // Swap Sign
        int ts = out_sign[j];
        out_sign[j] = out_sign[j + 1];
        out_sign[j + 1] = ts;
      }
    }
  }
}

/**
 * Calculates approximate distance using Linear Intersection.
 * Complexity: O(x) instead of O(x^2) because inputs are sorted by index.
 */
static type dist_approx_linear(const int *v_idx, const int *v_sign,
                               const int *w_idx, const int *w_sign, int x) {
  type result = 0.0;
  int i = 0;
  int j = 0;

  // Linear scan intersection (Merge Join logic)
  while (i < x && j < x) {
    int idx_v = v_idx[i];
    int idx_w = w_idx[j];

    if (idx_v < idx_w) {
      i++;
    } else if (idx_v > idx_w) {
      j++;
    } else {
      // Indices match!
      // Formula simplification:
      // (v+ w+) + (v- w-) - (v+ w-) - (v- w+)
      // == (sign_v == sign_w) ? +1 : -1
      if (v_sign[i] == w_sign[j]) {
        result += 1.0;
      } else {
        result -= 1.0;
      }
      i++;
      j++;
    }
  }
  return result;
}

static type euclidean_distance(const type *v, const type *w, int D) {
  type sum = 0.0;
// Helper for compiler vectorization (SIMD)
#pragma omp simd reduction(+ : sum)
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

  // 1. Pivot Selection
  input->P = (int *)malloc(h * sizeof(int));
  int step = N / h;
  for (int j = 0; j < h; j++) {
    input->P[j] = step * j;
  }

  // 2. Pre-Quantize the ENTIRE Dataset
  // This removes the heavy O(D) quantization from the predict loop.
  g_ds_idx = (int *)malloc(N * x * sizeof(int));
  g_ds_sign = (int *)malloc(N * x * sizeof(int));

#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    quantize_vector(&input->DS[i * D], D, x, &g_ds_idx[i * x],
                    &g_ds_sign[i * x]);
  }

  // 3. Pre-Quantize Pivots (into temp buffers for index construction)
  int *pivot_idxs = malloc(h * x * sizeof(int));
  int *pivot_signs = malloc(h * x * sizeof(int));

#pragma omp parallel for
  for (int j = 0; j < h; j++) {
    type *p_vec = &input->DS[input->P[j] * D];
    quantize_vector(p_vec, D, x, &pivot_idxs[j * x], &pivot_signs[j * x]);
  }

  // 4. Build Index Matrix
  input->index = (MATRIX)_mm_malloc(N * h * sizeof(type), align);

#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    // Use pre-quantized dataset point
    int *p_idx = &g_ds_idx[i * x];
    int *p_sign = &g_ds_sign[i * x];

    for (int j = 0; j < h; j++) {
      type dist = dist_approx_linear(p_idx, p_sign, &pivot_idxs[j * x],
                                     &pivot_signs[j * x], x);
      input->index[i * h + j] = dist;
    }
  }

  free(pivot_idxs);
  free(pivot_signs);
}

//  PREDICT (Querying)
void predict(params *input) {
  int N = input->N;
  int D = input->D;
  int h = input->h;
  int k = input->k;
  int x = input->x;
  int nq = input->nq;

  // We can clean up the global cache at the end of predict if this is a
  // one-shot run, or keep it if predict is called multiple times. Assuming
  // one-shot for this snippet structure.

#pragma omp parallel for
  for (int iq = 0; iq < nq; iq++) {
    // --- STACK ALLOCATION (VLA) ---
    // Much faster than malloc/free inside a loop
    int q_idx[x];
    int q_sign[x];
    int p_idx[x];
    int p_sign[x];
    type dist_query_pivots[h];

    // Local k-NN lists
    int current_knn_ids[k];
    type current_knn_dists[k];

    // Init k-NN
    for (int K = 0; K < k; K++) {
      current_knn_ids[K] = -1;
      current_knn_dists[K] = FLT_MAX;
    }
    type d_k_max = FLT_MAX; // Track max dist explicitly

    type *query_vec = &input->Q[iq * D];

    // 1. Quantize Query (Sorts indices internally)
    quantize_vector(query_vec, D, x, q_idx, q_sign);

    // 2. Calc Dist Query <-> Pivots
    for (int j = 0; j < h; j++) {
      // Retrieve pivot quantized data from Global Cache
      // (Pivots are just specific points in DS)
      int p_row = input->P[j];
      int *piv_idx_ptr = &g_ds_idx[p_row * x];
      int *piv_sign_ptr = &g_ds_sign[p_row * x];

      dist_query_pivots[j] =
          dist_approx_linear(q_idx, q_sign, piv_idx_ptr, piv_sign_ptr, x);
    }

    // 3. Scan Dataset
    for (int v = 0; v < N; v++) {
      // --- OPTIMIZATION: Early Exit on Lower Bound ---
      // Calculate Lower Bound (d_pvt*)
      type d_pvt_star = 0.0;

      // Check current worst distance
      if (current_knn_ids[k - 1] != -1) {
        // If buffer full, get actual max.
        // (Optimization: Maintain max in a variable or heap, but linear scan of
        // K is usually cheap)
        d_k_max = -1.0;
        for (int K = 0; K < k; K++)
          if (current_knn_dists[K] > d_k_max)
            d_k_max = current_knn_dists[K];
      } else {
        d_k_max = FLT_MAX;
      }

      // Calc Lower Bound loop
      int possible = 1;
      for (int j = 0; j < h; j++) {
        type d_vp = input->index[v * h + j];
        type diff = fabs(d_vp - dist_query_pivots[j]);
        if (diff > d_pvt_star)
          d_pvt_star = diff;

        // Early break if LB already exceeds K-th neighbor
        if (d_pvt_star >= d_k_max) {
          possible = 0;
          break;
        }
      }

      if (possible) {
        // --- CRITICAL OPTIMIZATION ---
        // Do NOT call quantize_vector here. Use pre-calculated global cache.
        int *v_idx_ptr = &g_ds_idx[v * x];
        int *v_sign_ptr = &g_ds_sign[v * x];

        type d_approx_qv =
            dist_approx_linear(q_idx, q_sign, v_idx_ptr, v_sign_ptr, x);

        if (d_approx_qv < d_k_max) {
          // Insert into k-NN (Simple linear replacement of max)
          // Find max pos
          int max_pos = -1;
          type current_max = -1.0;
          for (int K = 0; K < k; K++) {
            if (current_knn_dists[K] > current_max) {
              current_max = current_knn_dists[K];
              max_pos = K;
            }
          }
          // Replace
          if (max_pos != -1) {
            current_knn_ids[max_pos] = v;
            current_knn_dists[max_pos] = d_approx_qv;
          }
        }
      }
    }

    // 7. Refinement (Euclidean)
    for (int K = 0; K < k; K++) {
      int id = current_knn_ids[K];
      if (id != -1) {
        input->id_nn[iq * k + K] = id;
        input->dist_nn[iq * k + K] =
            euclidean_distance(query_vec, &input->DS[id * D], D);
      } else {
        input->id_nn[iq * k + K] = -1;
        input->dist_nn[iq * k + K] = FLT_MAX;
      }
    }
  }

  // Clean up static globals after all queries are done
  // (If the program continues after predict, handle this appropriately, e.g. a
  // separate cleanup function)
  if (g_ds_idx) {
    free(g_ds_idx);
    g_ds_idx = NULL;
  }
  if (g_ds_sign) {
    free(g_ds_sign);
    g_ds_sign = NULL;
  }
}
