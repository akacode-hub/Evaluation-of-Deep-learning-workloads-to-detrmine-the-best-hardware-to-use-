#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>

#define MIN_NUM 1
#define MAX_NUM 1000000

const int num_classes = 2;
const int num_blocks = 1;
const int num_threads_per_block = 1;
int tot_threads = num_blocks * num_threads_per_block;

__device__ int find_bin(int val, float * min_range_cls, float * max_range_cls){

    int i;
    
    for(i = 0; i < num_classes; i++) 
    {
        if(val>=min_range_cls[i] && val<max_range_cls[i])
            return  i;
    }

    return -1;
}

__global__ void hist_binning(int * data, int * hist_bin, int * cls_el, float * min_range_cls, float * max_range_cls, int * min_tidxs, int * max_tidxs)
{   
    int __shared__ hist_per_block[num_classes];

    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    
    int min_id, max_id;
    int i, cls_id;

    for (i = 0; i < num_classes; i += 1) 
        hist_per_block[i] = 0;

    __syncthreads();

    min_id = min_tidxs[tid];
    max_id = max_tidxs[tid];
    // printf("tid: %d, min_id: %d, max_id: %d\n", tid, min_id, max_id);

    for(i=min_id;i<max_id;i++){
        cls_id = find_bin(data[i], min_range_cls, max_range_cls);    
        hist_per_block[cls_id] += 1;
        cls_el[cls_id] = data[i];
        //printf("1 i: %d, val: %d, cls_id: %d\n", i, data[i], cls_id);
    }

    __syncthreads();
    
    for (i = 0; i < num_classes; i += 1) 
    {   
        hist_bin[i] += hist_per_block[i];
        printf("bid: %d, bblk: %d, tid: %d, hist %d: %d\n", blockIdx.x, blockDim.x, threadIdx.x, i, hist_bin[i]);
    }
}

void dist_data_tids(int * min_tidxs, int * max_tidxs, int data_len)
{

    int i;
    float interval = (float) data_len / (float) tot_threads;
    interval = floor(interval);

    for(i=0; i<tot_threads; i++){

        min_tidxs[i] = i*interval;
        max_tidxs[i] = (i+1)*interval;

        if(i == tot_threads-1){
            max_tidxs[i] = data_len;
        }

    }
}

void gen_data(int * data, int data_len)
{

    int i;
    for(i=0; i<data_len; i++){
        data[i] = rand() % MAX_NUM + MIN_NUM;
    }

}

void set_classes(float * min_range_cls, float * max_range_cls, int num_classes){
    
    int i;
    
    float range = MAX_NUM - MIN_NUM;
    float interval = (float) range / (float) num_classes;
    interval = ceil(interval);
    
    for(i=0; i<num_classes; i++){

        min_range_cls[i] = i*interval + MIN_NUM;

        if(i == num_classes-1){
            max_range_cls[i] = MAX_NUM + 1;
        }else{
            max_range_cls[i] = (i+1)*interval + MIN_NUM;
        }
    }
}

void print_data(int * data, float * min_range_cls, float * max_range_cls, int * min_tids, int * max_tids, int data_len, int num_classes){

    int i;

    // printf("Input data: \n");
    // for(i=0; i<data_len; i++){
    //     printf("%d ",data[i]);
    // }
    // printf("\n");

    printf("Class range: \n");
    for(i=0; i<num_classes; i++){
        printf("Class %d Min: %f, Max: %f\n", i, min_range_cls[i], max_range_cls[i]);
    }

    printf("Data ID range: \n");
    for(i=0; i<tot_threads; i++){
        printf("data %d Min: %d, Max: %d\n", i, min_tids[i], max_tids[i]);
    }

}

void print_results(int * hist_data, int * cls_el){

    int i;
    int sum = 0;
    for(i=0;i<num_classes;i++){
        sum += hist_data[i];
        printf("number of samples in cls %d: %d\n", i, hist_data[i]);
    }
    printf("histogram sum: %d\n", sum);

    for(i=0;i<num_classes;i++){
        printf("One element from cls %d: %d\n", i, cls_el[i]);
    }

}

int main(int argc, char *argv[])
{   
    srand(time(NULL));
    struct timespec start, end;
    int data_len;
    int * data, * data_gpu;
    int * data_cls_map, * data_cls_map_gpu;

    int * min_tidxs, * max_tidxs;
    float * min_range_cls, * max_range_cls;
    int * min_tidxs_gpu, * max_tidxs_gpu;
    float * min_range_cls_gpu, * max_range_cls_gpu;

    int * hist_bin, * hist_bin_gpu;
    int * cls_el, * cls_el_gpu;

    assert(("./Q1 <number of values>", argc == 2));

    data_len = atoi(argv[1]);

    min_range_cls = (float *)calloc(num_classes, sizeof(float));
    max_range_cls = (float *)calloc(num_classes, sizeof(float));
    cudaMalloc((void **) &min_range_cls_gpu, sizeof(float)*num_classes);
    cudaMalloc((void **) &max_range_cls_gpu, sizeof(float)*num_classes);

    min_tidxs = (int *)calloc(tot_threads, sizeof(int));
    max_tidxs = (int *)calloc(tot_threads, sizeof(int));
    cudaMalloc((void **) &min_tidxs_gpu, sizeof(int)*tot_threads);
    cudaMalloc((void **) &max_tidxs_gpu, sizeof(int)*tot_threads);

    hist_bin = (int *)calloc(num_classes, sizeof(int));
    cudaMalloc((void **) &hist_bin_gpu, sizeof(int)*num_classes);
    cudaMemset(hist_bin_gpu, 0, sizeof(int)*num_classes);

    cls_el = (int *)calloc(num_classes, sizeof(int));
    cudaMalloc((void **) &cls_el_gpu, sizeof(int)*num_classes);

    data = (int *)calloc(data_len, sizeof(int));
    gen_data(data, data_len);
    set_classes(min_range_cls, max_range_cls, num_classes);
    dist_data_tids(min_tidxs, max_tidxs, data_len);

    print_data(data, min_range_cls, max_range_cls, min_tidxs, max_tidxs, data_len, num_classes);

    cudaMalloc((void **) &data_gpu, sizeof(int)*data_len);
    
    cudaMemcpy(data_gpu, data, sizeof(int)*data_len, cudaMemcpyHostToDevice);   

    cudaMemcpy(min_range_cls_gpu, min_range_cls, sizeof(int)*num_classes, cudaMemcpyHostToDevice);
    cudaMemcpy(max_range_cls_gpu, max_range_cls, sizeof(int)*num_classes, cudaMemcpyHostToDevice);

    cudaMemcpy(min_tidxs_gpu, min_tidxs, sizeof(int)*tot_threads, cudaMemcpyHostToDevice);
    cudaMemcpy(max_tidxs_gpu, max_tidxs, sizeof(int)*tot_threads, cudaMemcpyHostToDevice);

    hist_binning<<<num_blocks, num_threads_per_block>>>(data_gpu, hist_bin_gpu, cls_el_gpu, min_range_cls_gpu, max_range_cls_gpu, min_tidxs_gpu, max_tidxs_gpu);

    cudaMemcpy(hist_bin, hist_bin_gpu, sizeof(int)*num_classes, cudaMemcpyDeviceToHost);
    cudaMemcpy(cls_el, cls_el_gpu, sizeof(int)*num_classes, cudaMemcpyDeviceToHost);

    print_results(hist_bin, cls_el);

    cudaFree(data_gpu); 
    cudaFree(min_range_cls_gpu);
    cudaFree(max_range_cls_gpu);
    cudaFree(min_tidxs_gpu);
    cudaFree(max_tidxs_gpu);
    cudaFree(hist_bin_gpu);
    cudaFree(cls_el_gpu);
}
