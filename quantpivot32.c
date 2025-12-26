#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <xmmintrin.h>  // _mm_malloc, _mm_free
#include "common.h"

// ---------- utility: error ----------
static void die(const char* msg) {
    fprintf(stderr, "Errore: %s\n", msg);
    exit(EXIT_FAILURE);
}

// ---------- quantizzazione: top-x per |v_i| ----------
static void quantize_topx(const type* v, int D, int x, int* out_idx, signed char* out_sign) {
    // Manteniamo una lista "top x" ordinata per valore assoluto crescente (pos 0 = più piccolo tra i top)
    // così sostituire è facile.
    // out_sign: +1 se v[idx]>=0, -1 se v[idx]<0
    // Inizializzazione con -inf
    type* best_abs = (type*)malloc((size_t)x * sizeof(type));
    if (!best_abs) die("malloc best_abs fallita");

    for (int t = 0; t < x; t++) {
        best_abs[t] = (type)-1.0f;
        out_idx[t] = -1;
        out_sign[t] = +1;
    }

    for (int i = 0; i < D; i++) {
        type a = (type)fabs((double)v[i]);
        if (a <= best_abs[0]) continue; // non entra nei top-x

        // inserisci a in posizione corretta (tenendo l'array crescente)
        int pos = 0;
        while (pos + 1 < x && a > best_abs[pos + 1]) pos++;

        // shift a sinistra fino a 0, buttando fuori il più piccolo
        for (int t = 0; t < pos; t++) {
            best_abs[t] = best_abs[t + 1];
            out_idx[t] = out_idx[t + 1];
            out_sign[t] = out_sign[t + 1];
        }

        best_abs[pos] = a;
        out_idx[pos] = i;
        out_sign[pos] = (v[i] >= (type)0) ? (signed char)+1 : (signed char)-1;
    }

    free(best_abs);

    // Nota: out_idx può contenere -1 se x>D (ma noi lo impediamo con controlli a monte)
}

// ---------- distanza approssimata tra due vettori quantizzati ----------
static inline type approx_dist_quantized(
    const int* idxA, const signed char* sA,
    const int* idxB, const signed char* sB,
    int x
) {
    // Somma signA*signB sulle dimensioni in comune.
    // Complessità O(x^2) (x tipicamente piccolo).
    int acc = 0;
    for (int i = 0; i < x; i++) {
        int ia = idxA[i];
        for (int j = 0; j < x; j++) {
            if (ia == idxB[j]) {
                acc += (int)sA[i] * (int)sB[j];  // +1 o -1
                break;
            }
        }
    }
    return (type)acc;
}

// ---------- ordina per distanza crescente (k piccolo: insertion sort) ----------
static void sort_by_dist(type* dist, int* ids, int k) {
    for (int i = 1; i < k; i++) {
        type keyd = dist[i];
        int keyi = ids[i];
        int j = i - 1;
        while (j >= 0 && dist[j] > keyd) {
            dist[j + 1] = dist[j];
            ids[j + 1] = ids[j];
            j--;
        }
        dist[j + 1] = keyd;
        ids[j + 1] = keyi;
    }
}


// ---------- distanza euclidea reale ----------
static inline type euclid_dist(const type* a, const type* b, int D) {
    double acc = 0.0;
    for (int i = 0; i < D; i++) {
        double d = (double)a[i] - (double)b[i];
        acc += d * d;
    }
    return (type)sqrt(acc);
}


// ---------- FIT ----------
void fit(params* input) {
    if (!input) die("input NULL");
    if (!input->DS) die("DataSet NULL");
    if (input->N <= 0 || input->D <= 0) die("N o D non validi");
    if (input->h <= 0 || input->h > input->N) die("h non valido (deve essere 1..N)");
    if (input->x <= 0 || input->x > input->D) die("x non valido (deve essere 1..D)");

    const int N = input->N;
    const int D = input->D;
    const int h = input->h;
    const int x = input->x;

    // 1) Selezione dei pivot (come da traccia: passo floor(N/h))
    if (!input->P) {
        input->P = (int*)malloc((size_t)h * sizeof(int));
        if (!input->P) die("malloc P fallita");
    }

    int step = N / h; // floor(N/h)
    if (step <= 0) step = 1; // caso limite, ma con h<=N step>=1

    for (int j = 0; j < h; j++) {
        int idx = step * j;
        if (idx >= N) idx = N - 1; // sicurezza
        input->P[j] = idx;
    }

    // 2) Alloca index: N x h (row-major)
    // index[v*h + j] = \tilde d(v, pivot_j)
    if (input->index) {
        _mm_free(input->index);
        input->index = NULL;
    }

    size_t idx_elems = (size_t)N * (size_t)h;
    input->index = (type*)_mm_malloc(idx_elems * sizeof(type), align);
    if (!input->index) die("_mm_malloc index fallita");

    // 3) Pre-quantizzazione dataset e pivot (per non rifarla N*h volte)
    int* q_idx_ds = (int*)malloc((size_t)N * (size_t)x * sizeof(int));
    signed char* q_sgn_ds = (signed char*)malloc((size_t)N * (size_t)x * sizeof(signed char));
    int* q_idx_pv = (int*)malloc((size_t)h * (size_t)x * sizeof(int));
    signed char* q_sgn_pv = (signed char*)malloc((size_t)h * (size_t)x * sizeof(signed char));
    if (!q_idx_ds || !q_sgn_ds || !q_idx_pv || !q_sgn_pv) die("malloc quantizzazione fallita");

    // quantizza DS
    for (int i = 0; i < N; i++) {
        const type* v = &input->DS[(size_t)i * (size_t)D];
        quantize_topx(v, D, x, &q_idx_ds[(size_t)i * (size_t)x], &q_sgn_ds[(size_t)i * (size_t)x]);
    }

    // quantizza pivot
    for (int j = 0; j < h; j++) {
        int pid = input->P[j];
        const type* p = &input->DS[(size_t)pid * (size_t)D];
        quantize_topx(p, D, x, &q_idx_pv[(size_t)j * (size_t)x], &q_sgn_pv[(size_t)j * (size_t)x]);
    }

    // 4) Costruzione indice: index[v][j] = approx_dist(v, pivot_j)
    for (int i = 0; i < N; i++) {
        const int* idxV = &q_idx_ds[(size_t)i * (size_t)x];
        const signed char* sV = &q_sgn_ds[(size_t)i * (size_t)x];

        for (int j = 0; j < h; j++) {
            const int* idxP = &q_idx_pv[(size_t)j * (size_t)x];
            const signed char* sP = &q_sgn_pv[(size_t)j * (size_t)x];

            input->index[(size_t)i * (size_t)h + (size_t)j] =
                approx_dist_quantized(idxV, sV, idxP, sP, x);
        }
    }

    free(q_idx_ds);
    free(q_sgn_ds);
    free(q_idx_pv);
    free(q_sgn_pv);

    if (!input->silent) {
        fprintf(stdout, "fit: creati %d pivot, index %d x %d\n", h, N, h);
    }
}


void predict(params* input) {
    if (!input) die("input NULL");
    if (!input->DS || !input->Q) die("DS o Q NULL");
    if (!input->P || !input->index) die("fit non eseguita (P o index NULL)");

    const int N  = input->N;
    const int D  = input->D;
    const int nq = input->nq;
    const int h  = input->h;
    const int k  = input->k;
    const int x  = input->x;

    if (N <= 0 || D <= 0 || nq <= 0) die("N/D/nq non validi");
    if (h <= 0 || h > N) die("h non valido");
    if (k <= 0 || k > N) die("k non valido");
    if (x <= 0 || x > D) die("x non valido");

    // alloca output se necessario
    if (!input->id_nn) {
        input->id_nn = (int*)malloc((size_t)nq * (size_t)k * sizeof(int));
        if (!input->id_nn) die("malloc id_nn fallita");
    }
    if (!input->dist_nn) {
        input->dist_nn = (type*)_mm_malloc((size_t)nq * (size_t)k * sizeof(type), align);
        if (!input->dist_nn) die("_mm_malloc dist_nn fallita");
    }

    // Pre-quantizzazione dei pivot (usando DS e P)
    int* pv_idx = (int*)malloc((size_t)h * (size_t)x * sizeof(int));
    signed char* pv_sgn = (signed char*)malloc((size_t)h * (size_t)x * sizeof(signed char));
    if (!pv_idx || !pv_sgn) die("malloc pv quant fallita");

    for (int j = 0; j < h; j++) {
        const type* p = &input->DS[(size_t)input->P[j] * (size_t)D];
        quantize_topx(p, D, x, &pv_idx[(size_t)j * (size_t)x], &pv_sgn[(size_t)j * (size_t)x]);
    }

    // buffer per ~d(q,p)
    type* dq_p = (type*)malloc((size_t)h * sizeof(type));
    if (!dq_p) die("malloc dq_p fallita");

    // buffer quantizzazione query
    int* q_idx = (int*)malloc((size_t)x * sizeof(int));
    signed char* q_sgn = (signed char*)malloc((size_t)x * sizeof(signed char));
    if (!q_idx || !q_sgn) die("malloc q quant fallita");

    // buffer quantizzazione v (solo quando serve)
    int* v_idx = (int*)malloc((size_t)x * sizeof(int));
    signed char* v_sgn = (signed char*)malloc((size_t)x * sizeof(signed char));
    if (!v_idx || !v_sgn) die("malloc v quant fallita");

    for (int qi = 0; qi < nq; qi++) {
        const type* q = &input->Q[(size_t)qi * (size_t)D];

        // 1) quantizza q e calcola ~d(q,p) per ogni pivot (linee 2-3 pseudocodice) :contentReference[oaicite:3]{index=3}
        quantize_topx(q, D, x, q_idx, q_sgn);
        for (int j = 0; j < h; j++) {
            dq_p[j] = approx_dist_quantized(
                q_idx, q_sgn,
                &pv_idx[(size_t)j * (size_t)x], &pv_sgn[(size_t)j * (size_t)x],
                x
            );
        }

        // 2) inizializza K-NN con +inf (linea 1) :contentReference[oaicite:4]{index=4}
        int*  best_id = &input->id_nn[(size_t)qi * (size_t)k];
        type* best_d  = &input->dist_nn[(size_t)qi * (size_t)k];

        for (int i = 0; i < k; i++) {
            best_id[i] = -1;
            best_d[i]  = (type)INFINITY;
        }

        // 3) scansiona dataset
        for (int vi = 0; vi < N; vi++) {
            // 3a) calcola bound con pivot: d*_pvt = max_p |~d(v,p) - ~d(q,p)| (linea 5) :contentReference[oaicite:5]{index=5}
            type dstar = 0;
            const type* rowIndex = &input->index[(size_t)vi * (size_t)h];
            for (int j = 0; j < h; j++) {
                type diff = (type)fabs((double)(rowIndex[j] - dq_p[j]));
                if (diff > dstar) dstar = diff;
            }

            // 3b) prendi dmax_k (linea 6) :contentReference[oaicite:6]{index=6}
            int imax = argmax_k(best_d, k);
            type dmax = best_d[imax];

            // 3c) pruning (linea 7) :contentReference[oaicite:7]{index=7}
            if (dstar < dmax) {
                // calcola ~d(q,v) (linea 8) :contentReference[oaicite:8]{index=8}
                const type* v = &input->DS[(size_t)vi * (size_t)D];
                quantize_topx(v, D, x, v_idx, v_sgn);
                type dqv = approx_dist_quantized(q_idx, q_sgn, v_idx, v_sgn, x);

                // se migliora, inserisci (linee 9-10) :contentReference[oaicite:9]{index=9}
                if (dqv < dmax) {
                    best_d[imax]  = dqv;
                    best_id[imax] = vi;
                }
            }
        }

        // 4) calcola distanza reale sui K trovati (linee 11-12) :contentReference[oaicite:10]{index=10}
        for (int i = 0; i < k; i++) {
            if (best_id[i] < 0) {
                best_d[i] = (type)INFINITY;
                continue;
            }
            const type* v = &input->DS[(size_t)best_id[i] * (size_t)D];
            best_d[i] = euclid_dist(q, v, D);
        }

        // 5) ordina per distanza crescente (comodo per output)
        sort_by_dist(best_d, best_id, k);
    }

    free(pv_idx);
    free(pv_sgn);
    free(dq_p);
    free(q_idx);
    free(q_sgn);
    free(v_idx);
    free(v_sgn);

    if (!input->silent) {
        fprintf(stdout, "predict: processate %d query, restituiti %d vicini ciascuna\n", nq, k);
    }
}