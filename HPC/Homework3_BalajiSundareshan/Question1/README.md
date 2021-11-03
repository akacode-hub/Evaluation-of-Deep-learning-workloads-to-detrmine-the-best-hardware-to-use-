## Question1

This repository contains C program for the dining philosopher problem. The modifications mentioned in the part a, b, c are integrated in this code.

## Compile instructions
* For compiling this C program without AVX,
    `g++ -fopt-info-vec-all Q1.c -o Q1_noavx -lm`

* For compiling this C program with AVX,
    `gcc -O3 -fopt-info-vec-all -ftree-vectorize -unroll-loops -g -mavx -march=native -mtune=native Q1.c -o Q1_avx -lm`

## Run instructions

* To run with avx, 

    `./Q1_avx <input> <num_terms>`

* To run without avx, 

    `./Q1_noavx <input> <num_terms>`

## Log
* Q1_avx.log - log with avx
* Q1_noavx.log - log without avx
