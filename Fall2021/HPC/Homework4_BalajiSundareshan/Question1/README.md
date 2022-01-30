## Question1

This repository contains the following script:
* Q1: This C program takes total number of darts to be thrown as input and approximately calculates the value of pi.

## Compile instructions
* For compiling C program for part1, run 

`mpicc Q1.c -o Q1 -lm`

## Run instructions

* To run C program on Discovery Cluster, run the below command in a bash script. The bash script used `Q1_bash.script` is provided.

* To run C program,

    `$SRUN mpirun -mca btl_base_warn_component_unused 0 Q1 <number of darts>`

## Log
* `Q1.out` provides sample log for part 1 where input data values and histogram values are printed.