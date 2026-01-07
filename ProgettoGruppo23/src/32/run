#!/bin/bash
for f in $(ls *.nasm); do
        nasm -f elf64 -DPIC $f;
done;
gcc -msse -O0 -fPIC -z noexecstack *.o main.c -o quantpivot -lm

./quantpivot
