#include<math.h>
#include <stdio.h>
#include<time.h>
#include <iostream>
#include <stdlib.h>
//#include "GpuTimer.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
using namespace std;

#define BLOCK_SIZE 512
#define NUM_BINS 4096
#define MAX_VAL 127
#define PRIVATE 4096

// histogramCPU computes the histogram of an input array on the CPU
void histogramCPU(unsigned int* input, unsigned int* bins, unsigned int numElems) {
    for (int i=0; i<numElems; i++) {
        if (bins[input[i]] < MAX_VAL) {
            bins[input[i]]++;
        }
    }
}

// histogramGPU computes the histogram of an input array on the GPU
__global__ void histogramGPU(unsigned int* input, unsigned int* bins, unsigned int numElems) {
    int tx = threadIdx.x; int bx = blockIdx.x;

    // compute global thread coordinates
    int i = (bx * blockDim.x) + tx;

    // create a private histogram copy for each thread block
    __shared__ unsigned int hist[PRIVATE];

    // each thread must initialize more than 1 location
    if (PRIVATE > BLOCK_SIZE) {
        for (int j=tx; j<PRIVATE; j+=BLOCK_SIZE) {
            if (j < PRIVATE) {
                hist[j] = 0;
            }
        }
    }
    // use the first `PRIVATE` threads of each block to init
    else {
        if (tx < PRIVATE) {
            hist[tx] = 0;
        }
    }
    // wait for all threads in the block to finish
    __syncthreads();

    // update private histogram
    if (i < numElems) {
        atomicAdd(&(hist[input[i]]), 1);
    }
    // wait for all threads in the block to finish
    __syncthreads();

    // each thread must update more than 1 location
    if (PRIVATE > BLOCK_SIZE) {
        for (int j=tx; j<PRIVATE; j+=BLOCK_SIZE) {
            if (j < PRIVATE) {
                atomicAdd(&(bins[j]), hist[j]);
            }
        }
    }
    // use the first `PRIVATE` threads to update final histogram
    else {
        if (tx < PRIVATE) {
            atomicAdd(&(bins[tx]), hist[tx]);
        }
    }
}

// saturateGPU caps the bin frequencies to a maximum value of 127
__global__ void saturateGPU(unsigned int* bins, unsigned int numBins) {
    // global thread coordinates
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i < numBins) {
        if (bins[i] > MAX_VAL) {
            bins[i] = MAX_VAL;
        }
    }
}

int main(void) {
    // timers
    //GpuTimer timer0, timer1, timer2, timer3, timer4;

    // data params
    int inputLength;
    unsigned int* hostBins_CPU;
    unsigned int* hostInput; unsigned int* hostBins;
    unsigned int* deviceInput; unsigned int* deviceBins;

    // ask the user to enter the length of the input vector
    printf("Please enter the length of the input array\n");
    scanf("%d", &inputLength);

    // determine
    size_t histoSize = NUM_BINS * sizeof(unsigned int);
    size_t inSize = inputLength * sizeof(unsigned int);

    // allocate host memory
    hostInput = (unsigned int*)malloc(inSize);
    hostBins = (unsigned int*)malloc(histoSize);
    hostBins_CPU = (unsigned int*)malloc(histoSize);

    // randomly initialize input array
    srand(clock());
    for (int i=0; i<inputLength; i++) {
        hostInput[i] = int((float)rand()*(NUM_BINS-1)/float(RAND_MAX));
    }
    // for (int i=0; i<inputLength; i++) {
    //     printf("%d, ", hostInput[i]);
    // }
    // printf("\n");

    // allocate device memory
    //timer0.Start();
    cudaMalloc((void**)&deviceInput, inSize);
    cudaMalloc((void**)&deviceBins, histoSize);
    cudaMemset(deviceBins, 0, histoSize);
    //timer0.Stop();
    //float d_t0 = timer0.Elapsed();
    //printf("Time to allocate memory on the device is: %f msecs.\n", d_t0);

    // host2device transfer
    //timer1.Start();
    cudaMemcpy(deviceInput, hostInput, inSize, cudaMemcpyHostToDevice);
    //timer1.Stop();
    //float d_t1 = timer1.Elapsed();
    //printf("Time to copy the input array from the host to the device is: %f msecs.\n", d_t1);

    // kernel launch
    dim3 threadPerBlock(BLOCK_SIZE, 1, 1);
    dim3 blockPerGrid(ceil(inputLength/(float)BLOCK_SIZE), 1, 1);
    //timer2.Start();
    histogramGPU<<<blockPerGrid, threadPerBlock>>>(deviceInput, deviceBins, inputLength);
    //timer2.Stop();
    //float d_t2 = timer2.Elapsed();
    //printf("Implemented CUDA code for basic histogram calculation ran in: %f msecs.\n", d_t2);

    // saturate the bins
    threadPerBlock.x = BLOCK_SIZE;
    blockPerGrid.x = ceil(NUM_BINS/(float)BLOCK_SIZE);
    //timer3.Start();
    saturateGPU<<<blockPerGrid, threadPerBlock>>>(deviceBins, NUM_BINS);
    //timer3.Stop();
    //float d_t3 = timer3.Elapsed();
    //printf("Implemented CUDA code for output saturation ran in: %f msecs.\n", d_t3);

    // device2host transfer
    //timer4.Start();
    cudaMemcpy(hostBins, deviceBins, histoSize, cudaMemcpyDeviceToHost);
    //timer4.Stop();
    //float d_t4 = timer4.Elapsed();
    //printf("Time to copy the resulting Histogram from the device to the host is: %f msecs.\n", d_t4);

    // initialize CPU histogram array to 0
    for (int i=0; i<NUM_BINS; i++) {
        hostBins_CPU[i] = 0;
    }

    // run the CPU version
    clock_t begin = clock();
    histogramCPU(hostInput, hostBins_CPU, inputLength);
    clock_t end = clock();

    // printf("CPU: \n");
    // for (int i=0; i<NUM_BINS; i++) {
    //     printf("%d, ", hostBins_CPU[i]);
    // }
    // printf("\n");
    // printf("GPU: \n");
    // for (int i=0; i<NUM_BINS; i++) {
    //     printf("%d, ", hostBins[i]);
    // }

    // calculate total time for CPU and GPU
    //double time_spent = (double)(end - begin) / CLOCKS_PER_SEC*1000;
    //float total_device_time = d_t0 + d_t1 + d_t2 + d_t3 + d_t4;
    //float speedup = (float)time_spent / total_device_time;
    //printf("Total CPU code ran in: %f msecs.\n", time_spent);
    //printf("Total GPU code ran in: %f msecs.\n", total_device_time);
    //printf("GPU Speedup: %f\n", speedup);

    // verify that results match
    for (int i=0; i<NUM_BINS; i++) {
        if (abs(int(hostBins_CPU[i] - hostBins[i])) > 0) {
            fprintf(stderr, "Result verification failed at element (%d)!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED\n");

    // release resources
    free(hostBins); free(hostBins_CPU); free(hostInput);
    cudaFree(deviceInput); cudaFree(deviceBins);

    return 0;
}
