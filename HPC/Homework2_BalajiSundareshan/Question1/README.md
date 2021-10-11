## Question1

This repository contains C program for the dining philosopher problem. The modifications mentioned in the part a, b, c are integrated in this code.

## Compile instructions
* For compiling this C program,

`gcc dining_philosopher.c -o dining_philosopher -lpthread`

## Run instructions

* To run the general case, 

    `./dining_philosopher <num_philosophers> <min_dur_microsec> <max_dur_microsec> <priority_philosopher_id> <extra_fork>`

    where 
    min_dur_microsec-max_dur_microsec: is the duration range in microseconds used for both thinking and eating.
    priority_philosopher_id: The philosopher id to whom we want to give priority
    extra_fork: Enable whether we need extra fork.

    Eg.
    `/dining_philosopher 5 50 100 -1 0` (extra fork disabled, priority disabled)

* To run for part a,  

    `/dining_philosopher 5 50 100 -1 1` (extra fork enabled, priority disabled)

* To run for part b,  

    `/dining_philosopher 5 100 100 2 -1` (extra fork disabled, priority enabled)

* To run for part c,  

    `/dining_philosopher 5 100 100 -1 0` (extra fork disabled, priority disabled)

## Log
* Q1.log - sample log for general case
* Q1_a.log - sample log for part a.
* Q1_b.log - sample log for part b.
* Q1_c.log - sample log for part c.