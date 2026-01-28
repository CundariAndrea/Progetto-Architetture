# ==========================================
# CONFIGURAZIONE GENERALE
# ==========================================
CC = gcc
ASM = nasm

# Flag del compilatore (quelli che usavi tu)
CFLAGS = -Wall -fopenmp -mavx -O0 -msse -fPIC -z noexecstack
LIBS = -lm

# File comuni necessari per entrambe le versioni
# ATTENZIONE: Qui NON mettiamo né quantpivot64.c né quantpivot64.o
COMMON_SRCS = main.c ../helper/*.c

# Argomenti per l'esecuzione (definiti una volta sola per evitare errori)
RUN_ARGS = -d ../../../../datasets/dataset_2000x256_64.ds2 \
           -q ../../../../datasets/query_2000x256_64.ds2 \
           -i ../../../results/results_ids_2000x8_k8_x64_64.ds2 \
           -e ../../../results/results_dst_2000x8_k8_x64_64.ds2 \
           -t ../../../../results/results_ids_2000x8_k8_x64_64.ds2 \
           -r ../../../../results/results_dst_2000x8_k8_x64_64.ds2 \
           -h 16 -k 8 -x 64

# Nomi degli eseguibili
TARGET_C = knn_c
TARGET_ASM = knn_asm

.PHONY: all clean build_c run_c build_asm run_asm

# Default: se scrivi solo 'make', compila tutto ma non esegue
all: build_c build_asm

# ==========================================
# 1. VERSIONE PURA IN C
# ==========================================

# Compila main + helper + quantpivot64.c
build_c:
	@echo "--- COMPILAZIONE VERSIONE C ---"
	$(CC) $(CFLAGS) -o $(TARGET_C) $(COMMON_SRCS) $(LIBS)
	
# Esegue la versione C (dipende da build_c)
run_c: build_c
	@echo "--- ESECUZIONE VERSIONE C ---"
	./$(TARGET_C) $(RUN_ARGS)

# ==========================================
# 2. VERSIONE IBRIDA (C + ASSEMBLY)
# ==========================================

build_asm:
	@echo "--- COMPILAZIONE VERSIONE ASSEMBLY ---"
	# 1. Assembla il file .nasm in .o
	$(ASM) -f elf64 -DPIC quantpivot64asm.nasm -o quantpivot64asm.o
	
	# 2. Linka tutto. 
	# NOTA: Aggiungiamo -DUSE_ASM_IMPL per nascondere il codice C duplicato
	$(CC) $(CFLAGS) -DUSE_ASM_IMPL -o $(TARGET_ASM) main.c ../helper/*.c quantpivot64asm.o $(LIBS)

run_asm: build_asm
	@echo "--- ESECUZIONE VERSIONE ASSEMBLY ---"
	./$(TARGET_ASM) $(RUN_ARGS)
# ==========================================
# PULIZIA
# ==========================================
clean:
	@echo "--- PULIZIA ---"
	rm -f $(TARGET_C) $(TARGET_ASM) *.o

