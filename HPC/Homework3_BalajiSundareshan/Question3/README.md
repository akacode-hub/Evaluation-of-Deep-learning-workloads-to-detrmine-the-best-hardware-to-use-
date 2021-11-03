## Question3

This repository contains C program for the dining philosopher problem using Pthreads and OpenMP.

## Compile instructions
* For compiling this C program using Pthread,

`gcc dining_philosopher_pthread.c -o dining_philosopher_pthread -lpthread`

* For compiling this C program using OpenMP,

`gcc dining_philosopher_omp.c -o dining_philosopher_omp -fopenmp`

## Run instructions

* To run OpenMP implementation, 

    `./dining_philosopher <num_philosophers> <min_dur_microsec> <max_dur_microsec>`

    where 
    min_dur_microsec-max_dur_microsec: is the duration range in microseconds used for both thinking and eating.

    Eg.
    `./dining_philosopher_omp 5 50 50`

* To run Pthread implementation, 

    `./dining_philosopher <num_philosophers> <min_dur_microsec> <max_dur_microsec>`

    where 
    min_dur_microsec-max_dur_microsec: is the duration range in microseconds used for both thinking and eating.

    Eg.
    `./dining_philosopher_pthread 5 50 50`

## Log
* dining_philosopher_pthread.log - pthread log
* dining_philosopher_omp.log - OpenMP log
