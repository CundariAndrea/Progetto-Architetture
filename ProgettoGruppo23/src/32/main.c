#include <stdio.h>
#include <stdlib.h>
#include <xmmintrin.h> // Header per le istruzioni SSE (necessario per _mm_malloc)
#include <math.h>      // Per sqrt, fabs (calcoli matematici)
#include "common.h"    // Definizione della struct 'params' e del tipo 'type' (float)

/* * Dichiarazione delle funzioni esterne.
 * Queste funzioni sono definite nel file dell'algoritmo (es. quantpivot32.c o quantpivot64omp.c).
 * - fit: costruisce l'indice e sceglie i pivot.
 * - predict: esegue le query per trovare i K vicini.
 */
void fit(params* input);
void predict(params* input);

// ---------- Utility: Gestione Errori ----------
// Funzione helper per stampare un messaggio di errore e terminare il programma immediatamente.
static void die(const char* msg) {
    fprintf(stderr, "Errore: %s\n", msg); // Stampa su Standard Error
    exit(EXIT_FAILURE);                   // Restituisce codice di errore al sistema operativo
}

// ---------- Funzioni di Verifica (Sanity Check) ----------

/*
 * Calcola la distanza Euclidea "reale" tra due vettori a e b di dimensione D.
 * Questa funzione serve SOLO per verificare se la distanza restituita da 'predict' è corretta.
 * Non viene usata per la ricerca vera e propria (che dovrebbe essere ottimizzata).
 */
static inline type euclid_dist(const type* a, const type* b, int D) {
    double acc = 0.0;
    for (int i = 0; i < D; i++) {
        double d = (double)a[i] - (double)b[i]; // Differenza tra componenti
        acc += d * d;                           // Accumula il quadrato della differenza
    }
    return (type)sqrt(acc); // Radice quadrata della somma
}

/*
 * Controlla la coerenza dei risultati (Consistency Check).
 * Verifica se la distanza che l'algoritmo 'predict' ha scritto in p->dist_nn
 * corrisponde effettivamente alla distanza reale calcolata ora con 'euclid_dist'.
 * * Parametri:
 * - p: puntatore ai parametri/dati
 * - qi: indice della query da controllare (es. 0 per la prima query)
 * - eps: margine di errore tollerato (epsilon) per confronti floating point
 */
static void check_dist_nn_consistency(const params* p, int qi, double eps) {
    // Puntatore al vettore della query corrente
    const type* q = &p->Q[(size_t)qi * (size_t)p->D];
    
    // Puntatori ai risultati trovati per questa query (ID e Distanze)
    const int* ids = &p->id_nn[(size_t)qi * (size_t)p->k];
    const type* dists = &p->dist_nn[(size_t)qi * (size_t)p->k];
 
    // Itera su tutti i K vicini trovati
    for (int i = 0; i < p->k; i++) {
        int id = ids[i]; // Indice del punto nel dataset

        // Controllo validità indice
        if (id < 0 || id >= p->N) {
            printf("[FAIL] qi=%d pos=%d id=%d (fuori range)\n", qi, i, id);
            continue;
        }

        // Recupera il vettore reale dal dataset usando l'ID trovato
        const type* v = &p->DS[(size_t)id * (size_t)p->D];
        
        // Ricalcola la distanza reale matematicamente
        type d = euclid_dist(q, v, p->D);
        
        // Calcola la differenza tra la distanza calcolata dall'algoritmo e quella reale
        double diff = fabs((double)d - (double)dists[i]);
 
        // Se la differenza è troppa, c'è un bug nell'algoritmo predict
        if (diff > eps) {
            printf("[FAIL] qi=%d pos=%d id=%d dist_nn=%f euclid=%f diff=%g\n",
                   qi, i, id, (double)dists[i], (double)d, diff);
        }
    }
}
 
// ---------- Loader Dataset/Query ----------
/*
 * Carica una matrice da file binario.
 * Formato atteso del file:
 * [int: numero righe] [int: numero colonne] [float...: dati grezzi]
 */
MATRIX load_data(const char* filename, int *n, int *d) {
    FILE* fp = fopen(filename, "rb"); // Apre in modalità lettura binaria
    if (!fp) {
        printf("'%s': bad data file name!\n", filename);
        exit(EXIT_FAILURE);
    }
 
    int rows = 0, cols = 0;
 
    // Legge i primi 4 byte: Numero di righe (N o nq)
    if (fread(&rows, sizeof(int), 1, fp) != 1)
        die("Errore lettura numero righe");
 
    // Legge i successivi 4 byte: Numero di colonne (D)
    if (fread(&cols, sizeof(int), 1, fp) != 1)
        die("Errore lettura numero colonne");
 
    // Alloca memoria allineata per i dati (importante per SSE/AVX)
    // _mm_malloc garantisce che l'indirizzo di memoria sia multiplo di 'align' (es. 16 byte)
    MATRIX data = (MATRIX)_mm_malloc((size_t)rows * (size_t)cols * sizeof(type), align);
    if (!data) die("_mm_malloc fallita");
 
    // Legge tutto il blocco di dati in un colpo solo
    size_t tot = (size_t)rows * (size_t)cols;
    if (fread(data, sizeof(type), tot, fp) != tot)
        die("Errore lettura dati matrice");
 
    fclose(fp);
 
    // Restituisce le dimensioni al chiamante tramite puntatori
    *n = rows;
    *d = cols;
    return data; // Restituisce il puntatore ai dati caricati
}

// ---------- Main ----------
int main(int argc, char** argv) {
    // Verifica che siano stati passati i nomi dei file
    if (argc < 3) {
        fprintf(stderr, "Uso: %s <dataset> <query>\n", argv[0]);
        return 1;
    }
 
    // Inizializza la struttura parametri a zero
    params p = {0};
 
    // 1. CARICAMENTO DATI
    // Carica il dataset e riempie p.N e p.D automaticamente leggendo l'header del file
    p.DS = load_data(argv[1], &p.N, &p.D);
 
    // Carica le query e riempie p.nq e p.D (sovrascrive p.D, ma dovrebbe essere identico)
    p.Q  = load_data(argv[2], &p.nq, &p.D);
 
    // 2. CONFIGURAZIONE PARAMETRI ALGORITMO
    p.h = 8;      // Numero di pivot da usare
    p.k = 5;      // Numero di vicini (K-NN) da trovare per ogni query
    p.x = 4;      // Numero di dimensioni su cui fare la quantizzazione (top-x)
    p.silent = 0; // Se 0, stampa output di debug/info. Se 1, sta zitto.
 
    // 3. FASE DI FIT (Training/Indexing)
    // Qui l'algoritmo sceglie i pivot e pre-calcola le distanze (indice)
    printf("Chiamo fit...\n");
    fit(&p);
    printf("fit OK\n");
 
    // 4. FASE DI PREDICT (Testing/Searching)
    // Qui l'algoritmo cerca i K vicini per tutte le query caricate
    printf("Chiamo predict...\n");
    predict(&p);
    printf("predict OK\n");

    // 5. VERIFICA RISULTATI
    // Controlla la coerenza matematica dei risultati per la PRIMA query (indice 0)
    // Usa una tolleranza di 1e-4 (0.0001) per errori di arrotondamento float
    check_dist_nn_consistency(&p, 0, 1e-4);
    
    // 6. STAMPA RISULTATI
    // Stampa a video gli ID e le Distanze dei K vicini trovati per la prima query
    printf("\nKNN prima query:\n");
    for (int i = 0; i < p.k; i++) {
        // id_nn e dist_nn sono array piatti, quindi l'accesso è sequenziale
        printf("id=%d dist=%f\n", p.id_nn[i], p.dist_nn[i]);
    }
 
    // 7. PULIZIA MEMORIA (Cleanup)
    // Libera la memoria allineata allocata con _mm_malloc
    _mm_free(p.DS);
    _mm_free(p.Q);
    _mm_free(p.index);   // Allocato dentro fit()
    _mm_free(p.dist_nn); // Allocato dentro predict()
    
    // Libera la memoria standard allocata con malloc
    free(p.P);           // Allocato dentro fit()
    free(p.id_nn);       // Allocato dentro predict()
 
    return 0;
}