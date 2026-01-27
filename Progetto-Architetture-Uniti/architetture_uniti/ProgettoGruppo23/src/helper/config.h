#ifndef CONFIG_H
#define CONFIG_H

// Struttura che contiene tutti i parametri di configurazione
typedef struct {
    char* dsfilename;
    char* queryfilename;
    char* out_idnn_filename;
    char* out_distnn_filename;
    char* test_resid_filename;
    char* test_resdst_filename;
    int h;
    int k;
    int x;
    int silent;
} Config;

/**
 * Effettua il parsing degli argomenti da linea di comando e popola la struct Config.
 * Stampa messaggi di errore e usage su stderr se necessario, ma non chiama exit().
 * RETURN: 0 (PARSE_OK) | 1 (PARSE_FAIL)
 */
int parse_arguments(int argc, char* argv[], Config* config);
void print_config(const Config* config);

#endif
