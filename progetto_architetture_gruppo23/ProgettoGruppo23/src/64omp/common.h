#ifndef QUANTPIVOT_COMMON
#define QUANTPIVOT_COMMON

#define	type	double
#define	align	32

#define	MATRIX		type*
#define	VECTOR		type*

typedef struct{
	// Variabili
	MATRIX DS; 					// dataset
	int* P;						// vettore contenente gli indici dei pivot
	MATRIX index;				// indice
	MATRIX Q;					// query
	int* id_nn;					// per ogni query point gli ID dei K-NN
	MATRIX dist_nn;				// per ogni query point le distanze dai K-NN
	int h;						// numero di pivot
	int k;						// numero di vicini
	int x;						// parametro x per la quantizzazione
	int N;						// numero di righe del dataset
	int D;						// numero di colonne/feature del dataset
	int nq;						// numero delle query
	int silent;					// modalità silenziosa
} params;

#endif
