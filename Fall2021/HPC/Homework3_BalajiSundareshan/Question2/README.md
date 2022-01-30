## Question2

This repository contains C program for calculating the number of numbers that are evenly divisible by 3, 7 or both between 1-10000 using pthreads and semaphores.

## Compile instructions
* For compiling this C program,

`gcc Q2.c -o Q2 -lpthread`

## Run instructions

* To run the general case, 

    `./Q2 <largest_number> <num_threads> div1 div2`
* For this question,
    For 8 threads,
     `./Q2 10000 8 3 7`  
    For 16 threads,
     `./Q2 10000 8 3 7`  

## Log
* Q2_8.log for 8 threads
* Q2_8.log for 16 threads
