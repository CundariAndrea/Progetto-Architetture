CFLAGS = -m32 -Wall -fopenmp -O3 -msse -fPIC -z noexecstack

TARGET = knn

.PHONY: build run clean

build:
	for f in *.nasm; do \
		nasm -f elf32 -DPIC $$f; \
	done
	gcc $(CFLAGS) -o $(TARGET) main.c *.o ../helper/*.c -lm

run:
	./$(TARGET) \
		-d ../../../../datasets/dataset_2000x256_32.ds2 \
		-q ../../../../datasets/query_2000x256_32.ds2 \
		-i ../../../results/results_ids_2000x8_k8_x64_32.ds2 \
		-e ../../../results/results_dst_2000x8_k8_x64_32.ds2 \
		-t ../../../../results/results_ids_2000x8_k8_x64_32.ds2 \
		-r ../../../../results/results_dst_2000x8_k8_x64_32.ds2 \
		-h 16 -k 8 -x 64

clean:
	rm -f $(TARGET)