#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <omp.h>
#include <cuda.h>
#include <curand_kernel.h>

#define MIN_NUM 1
#define MAX_NUM 100
const int N = 3;

__global__ void non_tile_compute(int b[][N][N], int a[][N][N])
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > 0 && i < N-1 && j > 0 && j < N-1 && k > 0 &&  k < N-1)
    {
        a[i][j][k]=0.8*(b[i-1][j][k]+b[i+1][j][k]+b[i][j-1][k]
                        + b[i][j+1][k]+b[i][j][k-1]+b[i][j][k+1]);
    }

}

__global__ void tile_compute(int * a, int * b)
{

}

int main(int argc, char *argv[])
{
    int * a, * b; 
    
    cudaMallocManaged(&a, N*N*N*sizeof(int));
    cudaMallocManaged(&b, N*N*N*sizeof(int));

    int b_vals[N][N][N];
    int i, j, k;

    // generate data
    for(k=0;k<N;k++){
        for(j=0;j<N;j++){
            for(i=0;i<N;i++){
                b_vals[k][j][i] = rand() % MAX_NUM + MIN_NUM;
            }
        }
    }

    // print data
    for(k=0;k<N;k++){
        for(j=0;j<N;j++){
            for(i=0;i<N;i++){
                printf("i: %d, j: %d, k: %d, val: %d\n", i, j, k, b_vals[k][j][i]);
            }
        }
    }

    int (*a_vals)[N][N] = reinterpret_cast<int (*)[N][N]>(a);
    memcpy(b, &b_vals[0][0][0], sizeof(b_vals));

    dim3 threadsPerBlock(3, 3, 3);
    dim3 numBlocks(3, 3, 3);
    non_tile_compute<<<numBlocks, threadsPerBlock>>>(
                                            reinterpret_cast<int (*)[N][N]>(b),
                                            a_vals);

    cudaDeviceSynchronize();

    memcpy(a, &a_vals[0][0][0], sizeof(a_vals));

    // print result
    for(i=0;i<N*N*N;i++){
        printf("i: %d, a: %d\n", i, a[i]);
    }

    return 0;
}
