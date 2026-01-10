CC = gcc

# Aggiunto -fopenmp qui sotto!
CFLAGS = -m64 -O3 -msse4.2 -Wall -fopenmp

LIBS = -lm

TARGET = main
SRCS = main.c quantpivot64.c
OBJS = $(SRCS:.c=.o)

# Parametri (Adatta ai tuoi file)
FILE_DS = dataset_2000x256_64.ds2
FILE_Q = query_2000x256_64.ds2

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LIBS)

%.o: %.c common.h
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) $(FILE_DS) $(FILE_Q)

clean:
	rm -f $(OBJS) $(TARGET)