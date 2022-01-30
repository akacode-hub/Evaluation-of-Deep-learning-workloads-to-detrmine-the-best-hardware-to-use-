## Question2

This repository contains two scripts:
* Q2_1: This C program takes number of values and number of classes as input and produces a histogram for (1/# classes) th of
the input data values in each node.
* Q2_2: This C program takes number of values and number of classes as input and each node produces values only for a single class of the input data values.

## Compile instructions
* For compiling C program for part1, run 

`mpicc Q2_1.c -o Q2_1 -lm`

* For compiling C program for part2, run 

`mpicc Q2_2.c -o Q2_2 -lm`

## Run instructions

* To run C program on Discovery Cluster, run the below command in a bash script. The bash script used `Q2_bash.script` is provided.

* To run C program for part1,

    `$SRUN mpirun -mca btl_base_warn_component_unused 0 Q2_1 <number of values> <number of classes>`

* To run C program for part2,

    `$SRUN mpirun -mca btl_base_warn_component_unused 0 Q2_2 <number of values> <number of classes>`

## Log
* `Q2_1_log_withprint.out` provides sample log for part 1 where input data values and histogram values are printed.
* `Q2_1_log.out` provides sample log for part 1 with just histogram.
* `Q2_2_log_withprint.out` provides sample log for part 2 where input data values and histogram values are printed.
* `Q2_2_log.out` provides sample log for part 2 with just histogram.