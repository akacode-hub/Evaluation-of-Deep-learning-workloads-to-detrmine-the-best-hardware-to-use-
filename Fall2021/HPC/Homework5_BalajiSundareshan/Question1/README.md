## Question1

This repository contains the following script:
* Q1: This CUDA program takes total number of elements and number of classes as input and creates a histogram for all the classes. The CPU implementation (OpenMP) is also present here.

## Compile instructions
* For compiling CUDA program along with OpenMP, run 

`nvcc -o Q1 Q1.cu -Xcompiler -fopenmp`

## Run instructions

* To run this program on Discovery Cluster, run the below command in a bash script. The bash script used `bash.sh` is provided.

* To run this program,

    `./Q1 <number of elements>`

## Log
* `Q1.log` provides sample log where histogram values are printed for both CPU and GPU implementation. One sample element from each class are also printed.