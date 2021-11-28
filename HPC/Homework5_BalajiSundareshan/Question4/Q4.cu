#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>

const double PI25DT = 3.141592653589793238462643;         /* 25-digit-PI*/
const int iteration = 2;
const int num_blocks = 1000;
const int num_threads_per_block = 256;

__global__ void fill_dart_count(int *in_dart_counts)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    long toss, num_toss_in;
    double x, y;
    
    curandState_t rng;
	curand_init(clock64(), tid, 0, &rng);

    //printf("num_darts_thread: %d\n", iteration);
    num_toss_in = 0;
    for (toss = 0; toss < iteration; toss++) {

		x = curand_uniform(&rng); // Random x position in [0,1]
		y = curand_uniform(&rng); // Random y position in [0,1]

        if(x*x + y*y <= 1.0){
            num_toss_in += 1;
        }
    }
    
    in_dart_counts[tid] = num_toss_in;

}

__global__ void count_darts(int *in_dart_counts, long *num_darts)
{
    int i;
    long sum = 0.0;
    for(i = 0; i<num_blocks*num_threads_per_block; i++){
        // printf("in_dart_counts %d: %d\n",i,in_dart_counts[i]);
        sum += in_dart_counts[i];
    }
    // printf("sum: %ld\n",sum);
    *num_darts = sum;    
}

int main(int argc, char *argv[])
{   
    srand(time(NULL));
    
    struct timespec start, end;

    assert(("./Q1 <number of darts>", argc == 2));
    long num_darts = atoi(argv[1]);
    assert(("num_darts should be divisible by 256*1000", num_darts % num_blocks * num_threads_per_block == 0));

    int num_dart_per_thread = num_darts / (num_blocks * num_threads_per_block);

    printf("Number of blocks: %d\n", num_blocks);
    printf("Number of threads per block: %d\n", num_threads_per_block);
    printf("Number of darts per thread: %d\n", num_dart_per_thread);
    printf("Total number of darts thrown: %d\n", num_darts);

    clock_gettime(CLOCK_MONOTONIC, &start);

    int *dart_counts_block;
    cudaMalloc((void**)&dart_counts_block, num_blocks * num_threads_per_block * sizeof(int));

    long *num_darts_gpu, num_darts_in;
    cudaMalloc((void**)&num_darts_gpu, sizeof(num_darts_in));

    int *num_dart_per_thread_gpu;
    cudaMalloc((void**)&num_dart_per_thread_gpu, sizeof(num_dart_per_thread));

    fill_dart_count<<<num_blocks, num_threads_per_block>>>(dart_counts_block);

    count_darts<<<1, 1>>>(dart_counts_block, num_darts_gpu);

    cudaMemcpy(&num_darts_in, num_darts_gpu, sizeof(num_darts_in), cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_elapsed = (end.tv_sec - start.tv_sec);
    time_elapsed += (end.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf("num_darts_in: %ld\n", num_darts_in);

    double calculated_pi = (4*num_darts_in)/((double) num_darts);
    printf("Calculated_pi: %.16f\n", calculated_pi);

    double error = fabs(calculated_pi - PI25DT);
    printf("Elapsed time = %f seconds \n", time_elapsed);    
    printf("Calculated Pi is %.16f, Error is %.16f\n", calculated_pi, error);

    return 0;
}