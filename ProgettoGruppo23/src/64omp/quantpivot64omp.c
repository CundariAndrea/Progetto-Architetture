/**************************************************************************************
* 
* CdL Magistrale in Ingegneria Informatica
* Corso di Architetture e Programmazione dei Sistemi di Elaborazione - a.a. 2025/26
* 
* Progetto dell'algoritmo Calcolo Approssimato dei K Vicini
* in linguaggio assembly x86-64 + SSE
* 
* F. Angiulli F. Fassetti, dicembre 2025
* 
**************************************************************************************/

/*
* 
* Software necessario per l'esecuzione:
* 
*    NASM (www.nasm.us)
*    GCC (gcc.gnu.org)
* 
* entrambi sono disponibili come pacchetti software 
* installabili mediante il packaging tool del sistema 
* operativo; per esempio, su Ubuntu, mediante i comandi:
* 
*    sudo apt-get install nasm
*    sudo apt-get install gcc
* 
* potrebbe essere necessario installare le seguenti librerie:
* 
*    sudo apt-get install lib64gcc-4.8-dev (o altra versione)
*    sudo apt-get install libc6-dev-i386
* 
* Per generare il file eseguibile:
* 
* nasm -f elf64 pst64.nasm && gcc -m64 -msse -O0 -no-pie sseutils64.o pst64.o pst64c.c -o pst64c -lm && ./pst64c $pars
* 
* oppure
* 
* ./runpst64
* 
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <immintrin.h>   // _mm_malloc, _mm_free
#include <omp.h>
#include "common.h"

/* =========================================================
   Cache globale privata (Opzione A) - NON serve cambiare common.h
   ========================================================= */

static int*         g_DS_qidx = NULL;   // N*x
static signed char* g_DS_qsgn = NULL;   // N*x
static int          g_cached_N = 0;
static int          g_cached_D = 0;
static int          g_cached_x = 0;
static const type*  g_cached_DS_ptr = NULL;

/* =========================================================
   Utility
   ========================================================= */

static void die(const char* msg) {
    fprintf(stderr, "Errore: %s\n", msg);
    exit(EXIT_FAILURE);
}

static inline signed char sgn_type(type v) {
    return (v >= (type)0) ? (signed char)+1 : (signed char)-1;
}

/* =========================================================
   Quantizzazione top-x SENZA malloc:
   - out_idx/out_sign: lunghezza x
   - work_abs: lunghezza x (buffer fornito dal chiamante)
   Caso speciale: x==D -> idx=0..D-1, sign=sign(v[i]) (O(D))
   ========================================================= */

static void quantize_topx_buf(
    const type* v, int D, int x,
    int* out_idx, signed char* out_sign,
    type* work_abs
) {
    if (x == D) {
        for (int i = 0; i < D; i++) {
            out_idx[i] = i;
            out_sign[i] = sgn_type(v[i]);
        }
        return;
    }

    // init: work_abs crescente, work_abs[0] è il minimo dei top-x
    for (int t = 0; t < x; t++) {
        work_abs[t] = (type)-1.0;
        out_idx[t]  = -1;
        out_sign[t] = +1;
    }

    for (int i = 0; i < D; i++) {
        type a = (type)fabs((double)v[i]);
        if (a <= work_abs[0]) continue;

        int pos = 0;
        while (pos + 1 < x && a > work_abs[pos + 1]) pos++;

        for (int t = 0; t < pos; t++) {
            work_abs[t] = work_abs[t + 1];
            out_idx[t]  = out_idx[t + 1];
            out_sign[t] = out_sign[t + 1];
        }

        work_abs[pos] = a;
        out_idx[pos]  = i;
        out_sign[pos] = sgn_type(v[i]);
    }
}

/* =========================================================
   Distanza approssimata tra due vettori quantizzati
   - se x==D: O(D), assume idx=0..D-1 per entrambi (generato da quantize_topx_buf)
   - altrimenti: O(x^2)
   ========================================================= */

static inline type approx_dist_quantized(
    const int* idxA, const signed char* sA,
    const int* idxB, const signed char* sB,
    int x, int D
) {
    if (x == D) {
        int acc = 0;
        for (int i = 0; i < D; i++) acc += (int)sA[i] * (int)sB[i];
        return (type)acc;
    }

    int acc = 0;
    for (int i = 0; i < x; i++) {
        int ia = idxA[i];
        if (ia < 0) continue;
        for (int j = 0; j < x; j++) {
            if (ia == idxB[j]) {
                acc += (int)sA[i] * (int)sB[j];
                break;
            }
        }
    }
    return (type)acc;
}

/* =========================================================
   Euclidea reale
   ========================================================= */

static inline type euclid_dist(const type* a, const type* b, int D) {
    double acc = 0.0;
    for (int i = 0; i < D; i++) {
        double d = (double)a[i] - (double)b[i];
        acc += d * d;
    }
    return (type)sqrt(acc);
}

/* =========================================================
   Helpers KNN
   ========================================================= */

static inline int argmax_k(const type* dist, int k) {
    int im = 0;
    for (int i = 1; i < k; i++) if (dist[i] > dist[im]) im = i;
    return im;
}

static void sort_by_dist(type* dist, int* ids, int k) {
    for (int i = 1; i < k; i++) {
        type keyd = dist[i];
        int  keyi = ids[i];
        int j = i - 1;
        while (j >= 0 && dist[j] > keyd) {
            dist[j + 1] = dist[j];
            ids[j + 1]  = ids[j];
            j--;
        }
        dist[j + 1] = keyd;
        ids[j + 1]  = keyi;
    }
}

/* =========================================================
   Facoltativo: libera la cache globale (chiamalo nel main se vuoi)
   ========================================================= */
void quantpivot_free_cache(void) {
    free(g_DS_qidx); g_DS_qidx = NULL;
    free(g_DS_qsgn); g_DS_qsgn = NULL;
    g_cached_N = g_cached_D = g_cached_x = 0;
    g_cached_DS_ptr = NULL;
}

/* =========================================================
   FIT
   - seleziona pivot
   - costruisce la cache globale DS_qidx/DS_qsgn (N*x) in parallelo
   - quantizza pivot (h*x)
   - costruisce index (N*h) in parallelo
   ========================================================= */
void fit(params* input) {
    if (!input) die("input NULL");
    if (!input->DS) die("DS NULL");
    if (input->N <= 0 || input->D <= 0) die("N o D non validi");
    if (input->h <= 0 || input->h > input->N) die("h non valido");
    if (input->x <= 0 || input->x > input->D) die("x non valido");

    const int N = input->N;
    const int D = input->D;
    const int h = input->h;
    const int x = input->x;

    // 1) pivot selection (step + clamp)
    if (!input->P) {
        input->P = (int*)malloc((size_t)h * sizeof(int));
        if (!input->P) die("malloc P fallita");
    }

    int step = N / h;
    if (step <= 0) step = 1;

    for (int j = 0; j < h; j++) {
        int idx = step * j;
        if (idx >= N) idx = N - 1;
        input->P[j] = idx;
    }

    // 2) index allocation N*h
    if (input->index) {
        _mm_free(input->index);
        input->index = NULL;
    }
    input->index = (type*)_mm_malloc((size_t)N * (size_t)h * sizeof(type), align);
    if (!input->index) die("_mm_malloc index fallita");

    // 3) (Re)alloc cache globale N*x
    //    Se cambi DS / N / D / x, la rigeneriamo.
    if (g_DS_qidx) { free(g_DS_qidx); g_DS_qidx = NULL; }
    if (g_DS_qsgn) { free(g_DS_qsgn); g_DS_qsgn = NULL; }

    g_DS_qidx = (int*)malloc((size_t)N * (size_t)x * sizeof(int));
    g_DS_qsgn = (signed char*)malloc((size_t)N * (size_t)x * sizeof(signed char));
    if (!g_DS_qidx || !g_DS_qsgn) die("malloc cache DS fallita");

    g_cached_N = N;
    g_cached_D = D;
    g_cached_x = x;
    g_cached_DS_ptr = input->DS;

    // 3a) quantizza DS in parallelo (solo parallel for)
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        // buffer locali (VLA): x elementi
        type absbuf[x];

        const type* v = &input->DS[(size_t)i * (size_t)D];
        quantize_topx_buf(
            v, D, x,
            &g_DS_qidx[(size_t)i * (size_t)x],
            &g_DS_qsgn[(size_t)i * (size_t)x],
            absbuf
        );
    }

    // 4) quantizza pivot (h*x) - serial (h spesso piccolo)
    int* pv_idx = (int*)malloc((size_t)h * (size_t)x * sizeof(int));
    signed char* pv_sgn = (signed char*)malloc((size_t)h * (size_t)x * sizeof(signed char));
    if (!pv_idx || !pv_sgn) die("malloc pv quant fallita");

    {
        type absbuf[x];
        for (int j = 0; j < h; j++) {
            const type* p = &input->DS[(size_t)input->P[j] * (size_t)D];
            quantize_topx_buf(
                p, D, x,
                &pv_idx[(size_t)j * (size_t)x],
                &pv_sgn[(size_t)j * (size_t)x],
                absbuf
            );
        }
    }

    // 5) costruisci index in parallelo (solo parallel for)
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        const int* idxV = &g_DS_qidx[(size_t)i * (size_t)x];
        const signed char* sV = &g_DS_qsgn[(size_t)i * (size_t)x];

        for (int j = 0; j < h; j++) {
            const int* idxP = &pv_idx[(size_t)j * (size_t)x];
            const signed char* sP = &pv_sgn[(size_t)j * (size_t)x];

            input->index[(size_t)i * (size_t)h + (size_t)j] =
                approx_dist_quantized(idxV, sV, idxP, sP, x, D);
        }
    }

    free(pv_idx);
    free(pv_sgn);

    if (!input->silent) {
        fprintf(stdout, "fit: creati %d pivot, index %d x %d (cache DS %d x %d)\n", h, N, h, N, x);
    }
}

/* =========================================================
   PREDICT
   - parallelo sulle query (parallel for)
   - usa cache globale g_DS_qidx/g_DS_qsgn (NO riquantizzazione DS)
   ========================================================= */
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
    if (k <= 0 || k > N) die("k non valido");

    // verifica cache valida
    if (!g_DS_qidx || !g_DS_qsgn ||
        g_cached_N != N || g_cached_D != D || g_cached_x != x ||
        g_cached_DS_ptr != input->DS)
    {
        die("cache DS non valida: chiama fit() con lo stesso DS/N/D/x prima di predict()");
    }

    // alloca output se necessario
    if (!input->id_nn) {
        input->id_nn = (int*)malloc((size_t)nq * (size_t)k * sizeof(int));
        if (!input->id_nn) die("malloc id_nn fallita");
    }
    if (!input->dist_nn) {
        input->dist_nn = (type*)_mm_malloc((size_t)nq * (size_t)k * sizeof(type), align);
        if (!input->dist_nn) die("_mm_malloc dist_nn fallita");
    }

    // quantizza pivot (h*x) una volta
    int* pv_idx = (int*)malloc((size_t)h * (size_t)x * sizeof(int));
    signed char* pv_sgn = (signed char*)malloc((size_t)h * (size_t)x * sizeof(signed char));
    if (!pv_idx || !pv_sgn) die("malloc pv quant fallita");

    {
        type absbuf[x];
        for (int j = 0; j < h; j++) {
            const type* p = &input->DS[(size_t)input->P[j] * (size_t)D];
            quantize_topx_buf(
                p, D, x,
                &pv_idx[(size_t)j * (size_t)x],
                &pv_sgn[(size_t)j * (size_t)x],
                absbuf
            );
        }
    }

    // parallelo sulle query (solo parallel for)
    #pragma omp parallel for
    for (int qi = 0; qi < nq; qi++) {
        // buffer locali per questa query
        int q_idx[x];
        signed char q_sgn[x];
        type q_abs[x];
        type dq_p[h];   // ~d(q,p) per ogni pivot

        const type* q = &input->Q[(size_t)qi * (size_t)D];

        // quantizza query
        quantize_topx_buf(q, D, x, q_idx, q_sgn, q_abs);

        // calcola ~d(q,p)
        for (int j = 0; j < h; j++) {
            dq_p[j] = approx_dist_quantized(
                q_idx, q_sgn,
                &pv_idx[(size_t)j * (size_t)x],
                &pv_sgn[(size_t)j * (size_t)x],
                x, D
            );
        }

        // init best
        int*  best_id = &input->id_nn[(size_t)qi * (size_t)k];
        type* best_d  = &input->dist_nn[(size_t)qi * (size_t)k];

        for (int i = 0; i < k; i++) {
            best_id[i] = -1;
            best_d[i]  = (type)FLT_MAX;
        }

        // scan DS
        for (int vi = 0; vi < N; vi++) {
            // bound con pivot: d* = max_j |index[vi,j] - dq_p[j]|
            type dstar = (type)0;
            const type* rowIndex = &input->index[(size_t)vi * (size_t)h];

            for (int j = 0; j < h; j++) {
                type diff = (type)fabs((double)(rowIndex[j] - dq_p[j]));
                if (diff > dstar) dstar = diff;
            }

            int imax = argmax_k(best_d, k);
            type dmax = best_d[imax];

            if (dstar < dmax) {
                // ~d(q,v) usando cache DS
                const int* idxV = &g_DS_qidx[(size_t)vi * (size_t)x];
                const signed char* sV = &g_DS_qsgn[(size_t)vi * (size_t)x];

                type dqv = approx_dist_quantized(q_idx, q_sgn, idxV, sV, x, D);

                if (dqv < dmax) {
                    best_d[imax]  = dqv;
                    best_id[imax] = vi;
                }
            }
        }

        // distanza reale sui K trovati
        for (int i = 0; i < k; i++) {
            int id = best_id[i];
            if (id < 0) {
                best_d[i] = (type)FLT_MAX;
                continue;
            }
            const type* v = &input->DS[(size_t)id * (size_t)D];
            best_d[i] = euclid_dist(q, v, D);
        }

        sort_by_dist(best_d, best_id, k);
    }

    free(pv_idx);
    free(pv_sgn);

    if (!input->silent) {
        fprintf(stdout, "predict: processate %d query, restituiti %d vicini ciascuna\n", nq, k);
    }
}