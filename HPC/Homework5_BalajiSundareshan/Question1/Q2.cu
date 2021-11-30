#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <omp.h>
#include <cuda.h>
#include <curand_kernel.h>

#define MIN_NUM 1
#define MAX_NUM 100
const int N = 10;

__global__ void non_tile_compute(int b[][N][N], int a[][N][N])
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < N && j < N && k < N)
        a[i][j][k] = b[i][j][k];

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

    for(k=0;k<N;k++){
        for(j=0;j<N;j++){
            for(i=0;i<N;i++){
                b_vals[k][j][i] = i + j + k;
            }
        }
    }

    int (*a_vals)[N][N] = reinterpret_cast<int (*)[N][N]>(a);
    memcpy(b, &b_vals[0][0][0], N*N*N);

    dim3 threadsPerBlock(1, 1, 1);
    dim3 numBlocks(1, 1, 1);
    non_tile_compute<<<numBlocks, threadsPerBlock>>>(
                                            reinterpret_cast<int (*)[N][N]>(b),
                                            a_vals);

    return 0;
}
