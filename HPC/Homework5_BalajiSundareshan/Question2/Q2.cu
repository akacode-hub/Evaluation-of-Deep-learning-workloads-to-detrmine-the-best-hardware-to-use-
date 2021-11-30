#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <omp.h>
#include <cuda.h>
#include <curand_kernel.h>

#define MIN_NUM 1
#define MAX_NUM 100
const int N = 4;
const int num_threads_per_block = 2;
const int num_blocks = 2;
const int sblock_size = num_threads_per_block + 2;

__global__ void non_tile_compute(float b[][N][N], float a[][N][N])
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

__global__ void tile_compute(float b[][N][N], float a[][N][N])
{   
    float __shared__ shared_b[sblock_size][sblock_size][sblock_size];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int offx, offy, offz;
    int offset = num_threads_per_block;

    for(offx=0; offx<sblock_size; offx+=2){
        for(offy=0; offy<sblock_size; offy+=2){
            for(offz=0; offz<sblock_size; offz+=2){
                shared_b[threadIdx.x + offx][threadIdx.y + offy][threadIdx.z + offz] = b[i][j][k];
                printf("i: %d, j: %d, k: %d, ox: %d, oy: %d, oz: %d\n", i, j, k, threadIdx.x + offx, threadIdx.y + offy, threadIdx.z + offz);
            }
        }
    }

    __syncthreads();

    if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0){
    printf("shared: \n\n");
    int i_, j_, k_;
    for(i_=0;i_<sblock_size;i_++){
        for(j_=0;j_<sblock_size;j_++){
            for(k_=0;i_<sblock_size;k_++){
                printf("%f, ", shared_b[i_][j_][k_]);
                }
            }
        }
    }

    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    int tz = threadIdx.z + 1;

    if (i > 0 && i < N-1 && j > 0 && j < N-1 && k > 0 &&  k < N-1)
    {
        a[i][j][k] = 0.8*(shared_b[tx-1][ty][tz] + shared_b[tx+1][ty][tz] + shared_b[tx][ty-1][tz]
                        + shared_b[tx][ty+1][tz] + shared_b[tx][ty][tz-1] + shared_b[tx][ty][tz+1]);
    }

}

void gen_mat(float arr[][N][N]){

    int i, j, k;
    for(i=0;i<N;i++){
        for(j=0;j<N;j++){
            for(k=0;k<N;k++){
                arr[i][j][k] = rand() % MAX_NUM + MIN_NUM;
                arr[i][j][k] = (float)rand()/(float)(RAND_MAX) * arr[i][j][k];
                //arr[i][j][k] = 1;
            }
        }
    }
}

void print_mat(float arr[][N][N]){

    int i, j, k;
    for(i=0;i<N;i++){
        for(j=0;j<N;j++){
            for(k=0;k<N;k++){
                printf("%f, ", arr[i][j][k]);
            }
        }
    }
    printf("\n\n");
}

void compare_mat(float arr1[][N][N], float arr2[][N][N]){

    int i, j, k;
    
    for(k=0;k<N;k++){
        for(j=0;j<N;j++){
            for(i=0;i<N;i++){
                if (arr1[i][j][k] != arr2[i][j][k]){
                    printf("Test failed !!!\n");
                    return;
                }
            }
        }
    }
    printf("Test passed !!!\n");
}

int main(int argc, char *argv[])
{
    float * a_tile, * a_nontile, * b; 
    
    cudaMallocManaged(&a_tile, N*N*N*sizeof(float));
    cudaMallocManaged(&a_nontile, N*N*N*sizeof(float));
    cudaMallocManaged(&b, N*N*N*sizeof(float));

    float b_vals[N][N][N];

    // generate data
    gen_mat(b_vals);

    // print data
    printf("Input matrix: \n");
    print_mat(b_vals);

    float (*a_vals_tile)[N][N] = reinterpret_cast<float (*)[N][N]>(a_tile);
    float (*a_vals_nontile)[N][N] = reinterpret_cast<float (*)[N][N]>(a_nontile);

    memcpy(b, &b_vals[0][0][0], sizeof(b_vals));

    dim3 threads_per_block(num_threads_per_block, num_threads_per_block, num_threads_per_block);
    dim3 blocks(num_blocks, num_blocks, num_blocks);

    non_tile_compute<<<blocks, threads_per_block>>>(reinterpret_cast<float (*)[N][N]>(b), a_vals_nontile);

    tile_compute<<<blocks, threads_per_block>>>(reinterpret_cast<float (*)[N][N]>(b), a_vals_tile);

    cudaDeviceSynchronize();

    // print result
    compare_mat(a_vals_tile, a_vals_nontile);

    printf("Tiled: \n");
    print_mat(a_vals_tile);

    printf("Non Tiled: \n");
    print_mat(a_vals_nontile);

    return 0;
}




// __syncthreads();