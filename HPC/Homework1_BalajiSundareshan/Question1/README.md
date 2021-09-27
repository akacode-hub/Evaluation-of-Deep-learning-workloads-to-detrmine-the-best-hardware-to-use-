## Question1

This repository contains 3 single-threaded benchmark programs. More information about the benchmark is present in the write-up.

## Compile instructions
* For compiling each benchmark, run 

`./linpack_bench.sh`

`./memory_test.sh`

`./sparse_test.sh`

## Run instructions
* To run linpack benchmark, run 

    `./bincpp/linpack_bench`

* To run memory benchmark, run 

    `./bincpp/memory_test <log_n_min> <log_n_max>`
    
    runs the program for sizes N = 2^log_n_min to 2^log_n_max

* To run sparse matrix multiplication benchmark, run 

    `./bincpp/sparse_test <matrix_minimum_dimension> <matrix_maximum_dimension> <step increment> <fraction of sparsity>`

    Eg. `./bincpp/sparse_test 100 500 100 0.5` runs benchmark for matrix dimensions from 100 * 100 to 500 * 500 in steps of 100 with 50% sparsity.

## Log
* Sample log is present in this repository for each benchmark.