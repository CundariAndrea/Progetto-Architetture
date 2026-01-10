#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>      // Per FLT_MAX (valore massimo float)
#include <xmmintrin.h>  // _mm_malloc, _mm_free (gestione memoria allineata)
#include "common.h"

// ---------- Utility: Gestione Errori ----------
// Stampa un messaggio su stderr e termina il processo.
static void die(const char* msg) {
    fprintf(stderr, "Errore: %s\n", msg);
    exit(EXIT_FAILURE);
}

// ---------- Quantizzazione: Top-x per |v_i| ----------
/*
 * Questa funzione è il cuore della compressione dei dati.
 * Invece di usare tutto il vettore di dimensione D, estrae solo le 'x' componenti
 * con il valore assoluto più grande (quelle che portano più informazione "energetica").
 *
 * Input:
 * - v: vettore originale (dimensione D)
 * - x: quante componenti salvare
 * Output:
 * - out_idx: indici delle x componenti (es. la componente 5, la 12, la 0...)
 * - out_sign: segno delle componenti (+1 o -1)
 */
static void quantize_topx(const type* v, int D, int x, int* out_idx, signed char* out_sign) {
    // best_abs serve per mantenere ordinati i valori assoluti delle top-x componenti trovate finora.
    // È ordinato in modo CRESCENTE: best_abs[0] è il più piccolo dei grandi (la soglia d'ingresso).
    type* best_abs = (type*)malloc((size_t)x * sizeof(type));
    if (!best_abs) die("malloc best_abs fallita");

    // Inizializzazione: riempiamo con -1 (nessun valore trovato)
    for (int t = 0; t < x; t++) {
        best_abs[t] = (type)-1.0f;
        out_idx[t] = -1;
        out_sign[t] = +1;
    }

    // Scansiona tutte le dimensioni del vettore originale
    for (int i = 0; i < D; i++) {
        type a = (type)fabs((double)v[i]); // Valore assoluto corrente

        // Se il valore corrente è più piccolo del più piccolo nella nostra lista top-x,
        // non ci interessa, lo scartiamo.
        if (a <= best_abs[0]) continue; 

        // Se siamo qui, 'a' merita di entrare nella lista top-x.
        // Troviamo la posizione corretta per mantenere l'array ordinato.
        int pos = 0;
        while (pos + 1 < x && a > best_abs[pos + 1]) pos++;

        // Shift a sinistra degli elementi più piccoli per fare spazio a quello nuovo.
        // L'elemento in posizione 0 (il più piccolo dei top) viene buttato fuori.
        for (int t = 0; t < pos; t++) {
            best_abs[t] = best_abs[t + 1];
            out_idx[t] = out_idx[t + 1];
            out_sign[t] = out_sign[t + 1];
        }

        // Inseriamo il nuovo valore
        best_abs[pos] = a;
        out_idx[pos] = i;
        out_sign[pos] = (v[i] >= (type)0) ? (signed char)+1 : (signed char)-1;
    }

    free(best_abs);
}

// ---------- Distanza Approssimata tra vettori quantizzati ----------
/*
 * Calcola una "distanza" (in realtà una similarità inversa) usando solo 
 * le top-x componenti salvate.
 * Logica: Se due vettori hanno componenti importanti negli stessi indici (idxA == idxB)
 * e con lo stesso segno, sono simili.
 */
static inline type approx_dist_quantized(
    const int* idxA, const signed char* sA,
    const int* idxB, const signed char* sB,
    int x
) {
    // Complessità O(x^2). Dato che x è piccolo (es. 16, 32), è molto veloce.
    int acc = 0;
    for (int i = 0; i < x; i++) {
        int ia = idxA[i];
        if (ia < 0) continue; // Salta slot vuoti (se D < x)
        
        // Cerca se l'indice ia è presente anche nel vettore B
        for (int j = 0; j < x; j++) {
            if (ia == idxB[j]) {
                // Se c'è match di indice, aggiunge +1 se i segni sono uguali, -1 se opposti.
                acc += (int)sA[i] * (int)sB[j];  
                break;
            }
        }
    }
    return (type)acc;
}

// ---------- Helper: Trova indice del massimo ----------
// Usato per trovare il "peggiore" tra i K vicini attuali (quello con distanza maggiore),
// che sarà il primo ad essere rimpiazzato se troviamo un punto più vicino.
static inline int argmax_k(const type* dist, int k) {
    int im = 0;
    for (int i = 1; i < k; i++) {
        if (dist[i] > dist[im]) im = i;
    }
    return im;
}

// ---------- Helper: Ordinamento (Insertion Sort) ----------
// Ordina i K risultati finali per distanza crescente.
// Dato che K è piccolo (es. 10, 100), insertion sort è efficiente.
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

// ---------- Distanza Euclidea Reale ----------
// Calcola la distanza vera usando tutte le D dimensioni. Costoso O(D).
static inline type euclid_dist(const type* a, const type* b, int D) {
    double acc = 0.0;
    for (int i = 0; i < D; i++) {
        double d = (double)a[i] - (double)b[i];
        acc += d * d;
    }
    return (type)sqrt(acc);
}

// =========================================================
//                  FASE 1: FIT (Training)
// =========================================================
/*
 * Prepara le strutture dati:
 * 1. Sceglie 'h' pivot dal dataset.
 * 2. Pre-calcola la distanza approssimata tra TUTTI i punti del dataset e i pivot.
 * Questo crea una matrice 'index' di dimensione N x h.
 */
void fit(params* input) {
    // Controlli di sicurezza sui parametri
    if (!input) die("input NULL");
    if (!input->DS) die("DataSet NULL");
    if (input->N <= 0 || input->D <= 0) die("N o D non validi");
    if (input->h <= 0 || input->h > input->N) die("h non valido (deve essere 1..N)");
    if (input->x <= 0 || input->x > input->D) die("x non valido (deve essere 1..D)");

    const int N = input->N;
    const int D = input->D;
    const int h = input->h;
    const int x = input->x;

    // --- 1) Selezione dei Pivot ---
    // Sceglie h punti equispaziati nel dataset come pivot.
    if (!input->P) {
        input->P = (int*)malloc((size_t)h * sizeof(int));
        if (!input->P) die("malloc P fallita");
    }

    int step = N / h;            
    if (step <= 0) step = 1;     

    for (int j = 0; j < h; j++) {
        int idx = step * j;
        if (idx >= N) idx = N - 1;
        input->P[j] = idx; // Salva l'indice del pivot
    }

    // --- 2) Allocazione Indice ---
    // Matrice N x h che conterrà le distanze approssimate Punti <-> Pivot
    if (input->index) {
        _mm_free(input->index);
        input->index = NULL;
    }

    size_t idx_elems = (size_t)N * (size_t)h;
    input->index = (type*)_mm_malloc(idx_elems * sizeof(type), align);
    if (!input->index) die("_mm_malloc index fallita");

    // --- 3) Pre-quantizzazione temporanea ---
    // Per calcolare le distanze approssimate, dobbiamo quantizzare sia dataset che pivot.
    // NOTA: In questa versione (32.c), questi dati vengono liberati alla fine di fit().
    int* q_idx_ds = (int*)malloc((size_t)N * (size_t)x * sizeof(int));
    signed char* q_sgn_ds = (signed char*)malloc((size_t)N * (size_t)x * sizeof(signed char));
    int* q_idx_pv = (int*)malloc((size_t)h * (size_t)x * sizeof(int));
    signed char* q_sgn_pv = (signed char*)malloc((size_t)h * (size_t)x * sizeof(signed char));
    if (!q_idx_ds || !q_sgn_ds || !q_idx_pv || !q_sgn_pv) die("malloc quantizzazione fallita");

    // Quantizza tutto il Dataset
    for (int i = 0; i < N; i++) {
        const type* v = &input->DS[(size_t)i * (size_t)D];
        quantize_topx(v, D, x,
                      &q_idx_ds[(size_t)i * (size_t)x],
                      &q_sgn_ds[(size_t)i * (size_t)x]);
    }

    // Quantizza i Pivot
    for (int j = 0; j < h; j++) {
        int pid = input->P[j];
        const type* p = &input->DS[(size_t)pid * (size_t)D];
        quantize_topx(p, D, x,
                      &q_idx_pv[(size_t)j * (size_t)x],
                      &q_sgn_pv[(size_t)j * (size_t)x]);
    }

    // --- 4) Costruzione Indice ---
    // Riempie la tabella: index[i][j] = DistanzaApprossimata(Punto_i, Pivot_j)
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

    // Libera la memoria temporanea.
    // ATTENZIONE: Questo obbligherà 'predict' a ri-quantizzare i dati al volo (lento!).
    free(q_idx_ds);
    free(q_sgn_ds);
    free(q_idx_pv);
    free(q_sgn_pv);

    if (!input->silent) {
        fprintf(stdout, "fit: creati %d pivot, index %d x %d\n", h, N, h);
    }
}

// =========================================================
//                  FASE 2: PREDICT (Search)
// =========================================================
void predict(params* input) {
    // Controlli preliminari
    if (!input) die("input NULL");
    if (!input->DS || !input->Q) die("DS o Q NULL");
    if (!input->P || !input->index) die("fit non eseguita (P o index NULL)");

    const int N  = input->N;
    const int D  = input->D;
    const int nq = input->nq;
    const int h  = input->h;
    const int k  = input->k;
    const int x  = input->x;

    // Allocazione buffer risultati (se non già fatto)
    if (!input->id_nn) {
        input->id_nn = (int*)malloc((size_t)nq * (size_t)k * sizeof(int));
        if (!input->id_nn) die("malloc id_nn fallita");
    }
    if (!input->dist_nn) {
        input->dist_nn = (type*)_mm_malloc((size_t)nq * (size_t)k * sizeof(type), align);
        if (!input->dist_nn) die("_mm_malloc dist_nn fallita");
    }

    // --- 1) Pre-quantizzazione Pivot (di nuovo) ---
    // Poiché 'fit' ha liberato la memoria, dobbiamo ri-quantizzare i pivot per usarli ora.
    int* pv_idx = (int*)malloc((size_t)h * (size_t)x * sizeof(int));
    signed char* pv_sgn = (signed char*)malloc((size_t)h * (size_t)x * sizeof(signed char));
    if (!pv_idx || !pv_sgn) die("malloc pv quant fallita");

    for (int j = 0; j < h; j++) {
        const type* p = &input->DS[(size_t)input->P[j] * (size_t)D];
        quantize_topx(p, D, x,
                      &pv_idx[(size_t)j * (size_t)x],
                      &pv_sgn[(size_t)j * (size_t)x]);
    }

    // Buffer temporanei
    type* dq_p = (type*)malloc((size_t)h * sizeof(type)); // Distanza Query <-> Pivot
    int* q_idx = (int*)malloc((size_t)x * sizeof(int));   // Query quantizzata
    signed char* q_sgn = (signed char*)malloc((size_t)x * sizeof(signed char));
    
    // Buffer per ri-quantizzare i punti del dataset "on demand"
    int* v_idx = (int*)malloc((size_t)x * sizeof(int));
    signed char* v_sgn = (signed char*)malloc((size_t)x * sizeof(signed char));

    if (!dq_p || !q_idx || !q_sgn || !v_idx || !v_sgn) die("malloc buffer fallita");

    // --- CICLO SU TUTTE LE QUERY ---
    for (int qi = 0; qi < nq; qi++) {
        const type* q = &input->Q[(size_t)qi * (size_t)D];

        // A) Quantizza la query corrente e calcola distanze con i pivot
        quantize_topx(q, D, x, q_idx, q_sgn);
        for (int j = 0; j < h; j++) {
            dq_p[j] = approx_dist_quantized(
                q_idx, q_sgn,
                &pv_idx[(size_t)j * (size_t)x],
                &pv_sgn[(size_t)j * (size_t)x],
                x
            );
        }

        // B) Inizializza la lista dei K vicini con "Infinito"
        int* best_id = &input->id_nn[(size_t)qi * (size_t)k];
        type* best_d  = &input->dist_nn[(size_t)qi * (size_t)k];
        for (int i = 0; i < k; i++) {
            best_id[i] = -1;
            best_d[i]  = (type)FLT_MAX;
        }

        // C) SCANSIONE DEL DATASET
        for (int vi = 0; vi < N; vi++) {
            // --- C1) Filtering (Pruning) con Disuguaglianza Triangolare ---
            // Calcoliamo un lower bound (dstar) usando le distanze precalcolate dai pivot.
            // |d(v, p) - d(q, p)| <= d(v, q)
            // Se la differenza delle distanze dai pivot è GRANDE, v e q devono essere lontani.
            type dstar = (type)0;
            const type* rowIndex = &input->index[(size_t)vi * (size_t)h];
            
            for (int j = 0; j < h; j++) {
                type diff = (type)fabs((double)(rowIndex[j] - dq_p[j]));
                if (diff > dstar) dstar = diff;
            }

            // Troviamo la distanza peggiore (più grande) attualmente nei nostri Top-K
            int imax = argmax_k(best_d, k);
            type dmax = best_d[imax];

            // Se dstar (minima distanza possibile stimata) è già peggiore del nostro peggior vicino,
            // possiamo scartare 'vi' senza fare calcoli costosi.
            if (dstar < dmax) {
                // --- C2) Calcolo Distanza Approssimata ---
                // Il candidato 'vi' potrebbe essere buono. Calcoliamo la distanza approssimata.
                
                // NOTA PERFORMANCE: Qui avviene la ri-quantizzazione del punto 'vi'.
                // È un'operazione costosa O(D) che si ripete spesso, rallentando l'esecuzione.
                const type* v = &input->DS[(size_t)vi * (size_t)D];
                quantize_topx(v, D, x, v_idx, v_sgn);
                
                type dqv = approx_dist_quantized(q_idx, q_sgn, v_idx, v_sgn, x);

                // Se anche la distanza approssimata è promettente, salviamo il candidato
                // (per ora salviamo la distanza approssimata, dopo la correggeremo).
                if (dqv < dmax) {
                    best_d[imax]  = dqv;
                    best_id[imax] = vi;
                }
            }
        }

        // --- D) Refinement (Calcolo Distanza Reale) ---
        // Ora abbiamo i K "presunti" migliori candidati basati sulle approssimazioni.
        // Calcoliamo la vera distanza Euclidea per questi pochi sopravvissuti.
        for (int i = 0; i < k; i++) {
            if (best_id[i] < 0) {
                best_d[i] = (type)FLT_MAX;
                continue;
            }
            const type* v = &input->DS[(size_t)best_id[i] * (size_t)D];
            best_d[i] = euclid_dist(q, v, D); // Calcolo costoso O(D), ma solo su K punti.
        }

        // --- E) Ordinamento Finale ---
        sort_by_dist(best_d, best_id, k);
    }

    // Pulizia memoria locale
    free(pv_idx); free(pv_sgn);
    free(dq_p);
    free(q_idx); free(q_sgn);
    free(v_idx); free(v_sgn);

    if (!input->silent) {
        fprintf(stdout, "predict: processate %d query, restituiti %d vicini ciascuna\n", nq, k);
    }
}