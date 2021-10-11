## Question2

This repository contains two scripts:
* count_primes_pthreads: This C program takes an input number and number of threads and counts the total number of primes and prints them using pthreads.
* count_primes_omp: This C program takes an input number and number of threads and counts the total number of primes and prints them using OpenMP.

## Compile instructions
* For compiling pthread based C program, run 

`gcc count_primes_pthreads.c -o count_primes_pthreads -lpthread`

* For compiling OpenMP based C program, run 

`gcc count_primes_omp.c -o count_primes_omp -fopenmp`

## Run instructions
* To run pthread based C program, run 

    `./count_primes_pthreads <largest number> num_threads`

* To run OpenMP based C program, run 

    `./count_primes_omp <largest number> num_threads`
    
    runs the program for sizes N = 2^log_n_min to 2^log_n_max

## Log
* Sample log is present in this repository for both of these C programs.