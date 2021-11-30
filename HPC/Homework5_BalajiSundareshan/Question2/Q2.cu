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

void gen_mat(int arr[][N][N]){

    int i, j, k;
    for(k=0;k<N;k++){
        for(j=0;j<N;j++){
            for(i=0;i<N;i++){
                arr[k][j][i] = rand() % MAX_NUM + MIN_NUM;
            }
        }
    }
}

void print_mat(int arr[][N][N]){

    int i, j, k;
    for(k=0;k<N;k++){
        for(j=0;j<N;j++){
            for(i=0;i<N;i++){
                printf("i: %d, j: %d, k: %d, val: %d\n", i, j, k, arr[k][j][i]);
            }
        }
    }
}

int main(int argc, char *argv[])
{
    int * a, * b; 
    
    cudaMallocManaged(&a, N*N*N*sizeof(int));
    cudaMallocManaged(&b, N*N*N*sizeof(int));

    int b_vals[N][N][N];

    // generate data
    gen_mat(b_vals);

    // print data
    printf("Input matrix: \n");
    print_mat(b_vals);

    int (*a_vals)[N][N] = reinterpret_cast<int (*)[N][N]>(a);
    memcpy(b, &b_vals[0][0][0], sizeof(b_vals));

    dim3 threadsPerBlock(3, 3, 3);
    dim3 numBlocks(3, 3, 3);
    non_tile_compute<<<numBlocks, threadsPerBlock>>>(
                                            reinterpret_cast<int (*)[N][N]>(b),
                                            a_vals);

    cudaDeviceSynchronize();

    // print result
    printf("Result matrix: \n");
    print_mat(a_vals);   

    return 0;
}