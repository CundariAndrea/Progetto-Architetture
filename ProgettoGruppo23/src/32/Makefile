# ==========================================
# Makefile per KNN Quantizzato (32-bit)
# ==========================================

# Compilatore
CC = gcc

# Flag di compilazione:
# -m32       : Forza architettura 32 bit (richiesto dalla tua specifica)
# -O3        : Massima ottimizzazione
# -msse4.2   : Abilita istruzioni SSE (usate in _mm_malloc, etc.)
# -Wall      : Mostra tutti i warning
CFLAGS = -m32 -O3 -msse4.2 -Wall

# Librerie da linkare (-lm per math.h)
LIBS = -lm

# Nomi dei file
TARGET = knn32
SRCS = main.c quantpivot32.c
OBJS = $(SRCS:.c=.o)

# --- Parametri di esecuzione (Modificali qui se necessario) ---
FILE_DS    = dataset_2000x256_32.ds2
FILE_QUERY = query_2000x256_32.ds2
N          = 10000
D          = 256
K          = 10

# Regola di default (compila tutto)
all: $(TARGET)

# Regola per creare l'eseguibile (link)
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LIBS)

# Regola per compilare i file oggetto (.c -> .o)
%.o: %.c common.h
	$(CC) $(CFLAGS) -c $< -o $@

# Regola per eseguire il test
run: $(TARGET)
	@echo "--- Avvio esecuzione a 32 bit ---"
	./$(TARGET) $(FILE_DS) $(FILE_QUERY) $(N) $(D) $(K)

# Pulizia dei file compilati
clean:
	rm -f $(OBJS) $(TARGET)
	
