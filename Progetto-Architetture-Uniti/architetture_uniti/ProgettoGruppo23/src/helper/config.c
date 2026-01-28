#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <getopt.h>
#include "config.h"

static int file_exists(const char *filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

static void print_usage(const char *prog_name) {
    fprintf(stderr, "Uso: %s [opzioni]\n\n", prog_name);
    fprintf(stderr, "Nota: Tutti i parametri (eccetto -s) sono OBBLIGATORI.\n\n");
    fprintf(stderr, "Opzioni disponibili:\n");
    fprintf(stderr, "  -d, --dataset <path>   Imposta dsfilename\n");
    fprintf(stderr, "  -q, --query <path>     Imposta queryfilename\n");
    fprintf(stderr, "  -i, --idnn <path>      Imposta nome file output per idnn\n");
    fprintf(stderr, "  -e, --distnn <path>    Imposta nome file output per distnn\n");
    fprintf(stderr, "  -t, --idtest <path>    Imposta nome file degli ID pre-calcolati\n");
    fprintf(stderr, "  -t, --dsttest <path>   Imposta nome file delle DIST pre-calcolate\n");
    fprintf(stderr, "  -h <int>               Imposta parametro h (positivo)\n");
    fprintf(stderr, "  -k <int>               Imposta parametro k (positivo)\n");
    fprintf(stderr, "  -x <int>               Imposta parametro x (positivo)\n");
    fprintf(stderr, "  -s                     Attiva la modalità silent (opzionale)\n");
}

int parse_arguments(int argc, char* argv[], Config* config) {
    config->dsfilename = NULL;
    config->queryfilename = NULL;
    config->out_idnn_filename = NULL;
    config->out_distnn_filename = NULL;
    config->test_resid_filename = NULL;
    config->test_resdst_filename = NULL;
    config->h = -1;
    config->k = -1;
    config->x = -1;
    config->silent = 0; 

    int opt;
    int option_index = 0;

    static struct option long_options[] = {
        {"dataset", required_argument, 0, 'd'},
        {"query",   required_argument, 0, 'q'},
        {"idnn",  required_argument, 0, 'i'},
        {"distnn",  required_argument, 0, 'e'},
        {"idtest",  required_argument, 0, 't'},
        {"dsttest",  required_argument, 0, 'r'},
        {0, 0, 0, 0}
    };

    optind = 1; // Reset di optind

    while ((opt = getopt_long(argc, argv, "d:q:i:e:t:r:h:k:x:s", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'd': config->dsfilename = optarg; break;
            case 'q': config->queryfilename = optarg; break;
            case 'i': config->out_idnn_filename = optarg; break;
            case 'e': config->out_distnn_filename = optarg; break;
            case 't': config->test_resid_filename = optarg; break;
            case 'r': config->test_resdst_filename = optarg; break;
            case 'h': config->h = atoi(optarg); break;
            case 'k': config->k = atoi(optarg); break;
            case 'x': config->x = atoi(optarg); break;
            case 's': config->silent = 1; break;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    int missing = 0;
    if (!config->dsfilename) { fprintf(stderr, "Errore: Manca --dataset (-d)\n"); missing = 1; }
    if (!config->queryfilename) { fprintf(stderr, "Errore: Manca --query (-q)\n"); missing = 1; }
    if (!config->out_idnn_filename) { fprintf(stderr, "Errore: Manca --idnn (-i)\n"); missing = 1; }
    if (!config->out_distnn_filename) { fprintf(stderr, "Errore: Manca --distnn (-e)\n"); missing = 1; }
    if (!config->test_resid_filename) { fprintf(stderr, "Errore: Manca --idtest (-t)\n"); missing = 1; }
    if (!config->test_resdst_filename) { fprintf(stderr, "Errore: Manca --dsttest (-t)\n"); missing = 1; }
    if (config->h == -1) { fprintf(stderr, "Errore: Manca -h\n"); missing = 1; }
    if (config->k == -1) { fprintf(stderr, "Errore: Manca -k\n"); missing = 1; }
    if (config->x == -1) { fprintf(stderr, "Errore: Manca -x\n"); missing = 1; }

    if (missing) {
        fprintf(stderr, "\n");
        print_usage(argv[0]);
        return 1;
    }

    int val_error = 0;

    if (config->h <= 0) { fprintf(stderr, "Errore: 'h' deve essere positivo.\n"); val_error = 1; }
    if (config->k <= 0) { fprintf(stderr, "Errore: 'k' deve essere positivo.\n"); val_error = 1; }
    if (config->x <= 0) { fprintf(stderr, "Errore: 'x' deve essere positivo.\n"); val_error = 1; }

    if (!file_exists(config->dsfilename)) {
        fprintf(stderr, "Errore: Dataset '%s' non trovato.\n", config->dsfilename);
        val_error = 1;
    }
    if (!file_exists(config->queryfilename)) {
        fprintf(stderr, "Errore: Query file '%s' non trovato.\n", config->queryfilename);
        val_error = 1;
    }
    if (!file_exists(config->test_resid_filename)) {
        fprintf(stderr, "Errore: Result file ID '%s' non trovato.\n", config->test_resid_filename);
        val_error = 1;
    }
    if (!file_exists(config->test_resdst_filename)) {
        fprintf(stderr, "Errore: Result file DST '%s' non trovato.\n", config->test_resdst_filename);
        val_error = 1;
    }

    if (val_error) {
        return 1;
    }

    return 0;
}

void print_config(const Config* config) {
    printf("\n");
    printf("   ===================================================\n");
    printf("   |              QUANTPIVOT CONFIGURATION           |\n");
    printf("   ===================================================\n");
    printf("   | %-20s : %-24s |\n", "Dataset", config->dsfilename);
    printf("   | %-20s : %-24s |\n", "Query Set", config->queryfilename);
    printf("   |-------------------------------------------------|\n");
    printf("   | %-20s : %-24s |\n", "Output ID (NN)", config->out_idnn_filename);
    printf("   | %-20s : %-24s |\n", "Output Dist (NN)", config->out_distnn_filename);
    printf("   |-------------------------------------------------|\n");
    printf("   | %-20s : %-24s |\n", "Test file ID", config->test_resid_filename);
    printf("   | %-20s : %-24s |\n", "Test file Dist", config->test_resdst_filename);
    printf("   |-------------------------------------------------|\n");
    printf("   | %-20s : %-4d                     |\n", "Pivots (h)", config->h);
    printf("   | %-20s : %-4d                     |\n", "Neighbors (k)", config->k);
    printf("   | %-20s : %-4d                     |\n", "Quantization (x)", config->x);
    printf("   ===================================================\n\n");
}
