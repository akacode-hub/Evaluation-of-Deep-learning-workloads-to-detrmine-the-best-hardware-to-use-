## Question2

This repository contains the following script:
* Q2: This CUDA program generates a 64 * 64 * 64 matrix and does stencil computation. Both tiled and non-tiled versions are implemented here.

## Compile instructions
* For compiling CUDA program, run 

`nvcc -o Q2 Q2.cu`

## Run instructions

* To run this program on Discovery Cluster, run the below command in a bash script. The bash script used `bash.sh` is provided.

* To run this program,

    `./Q2`

## Log
* `Q2.log` provides sample log where the calculated matrix from both tiled and non-tiled implementation are compared with serial implementation for accuracy.