#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>

#define MIN_NUM 1
#define MAX_NUM 1000000
const int num_blocks = 10;
const int num_threads_per_block = 8;
int tot_threads = num_blocks * num_threads_per_block;

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
    data = (int *)calloc(data_len, sizeof(int));

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

int main(int argc, char *argv[])
{   
    srand(time(NULL));
    struct timespec start, end;
    int data_len, num_classes;
    int * data, * data_gpu;
    int * min_tidxs, * max_tidxs;
    int * hist_bins;
    float * min_range_cls, * max_range_cls;

    assert(("./Q1 <number of values> <number of classes>", argc == 3));

    data_len = atoi(argv[1]);
    num_classes = atoi(argv[2]);

    min_range_cls = (float *)calloc(num_classes, sizeof(float));
    max_range_cls = (float *)calloc(num_classes, sizeof(float));
    min_tidxs = (int *)calloc(tot_threads, sizeof(int));
    max_tidxs = (int *)calloc(tot_threads, sizeof(int));
    

    gen_data(data, data_len);
    set_classes(min_range_cls, max_range_cls, num_classes);
    dist_data_tids(min_tidxs, max_tidxs, data_len);

    print_data(data, min_range_cls, max_range_cls, min_tidxs, max_tidxs, data_len, num_classes);

    cudaMalloc((void **) &data_gpu, sizeof(int)*data_len);
    cudaMemcpy(data_gpu, data, sizeof(int)*data_len, cudaMemcpyHostToDevice);

}