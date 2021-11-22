#include <stdbool.h>
#include <mpi.h>  
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define MIN_NUM 1
#define MAX_NUM 1000

int * data;
int * local_data;
int * classes;
int * local_classes;
float * min_range_cls;
float * max_range_cls;

void gen_data(int global_data_len)
{

    unsigned int seed = (unsigned) time(NULL);
	srand(seed);

    data = (int *)calloc(global_data_len, sizeof(int));

    int i;
    for(i=0; i<global_data_len; i++){
        data[i] = rand() % MAX_NUM + MIN_NUM;
    }

}

void set_classes(int num_classes){

    local_classes = (int *)calloc(num_classes, sizeof(int));
    min_range_cls = (float *)calloc(num_classes, sizeof(float));
    max_range_cls = (float *)calloc(num_classes, sizeof(float));
    
    int i;
    
    float range = MAX_NUM - MIN_NUM;
    float interval = (float) range / (float) num_classes;
    interval = ceil(interval);

    for(i=0; i<num_classes; i++){

        min_range_cls[i] = i*interval + MIN_NUM;
        max_range_cls[i] = (i+1)*interval + MIN_NUM;
        local_classes[i] = 0;
    }

}

void print_data(int global_data_len, int num_classes){

    int i;

    printf("Input data: ");
    for(i=0; i<global_data_len; i++){
        printf("%d, ",data[i]);
    }
    printf("\n");

    for(i=0; i<num_classes; i++){
        printf("Min: %f, Max: %f\n", min_range_cls[i], max_range_cls[i]);
    }

}

void print_hist(int num_classes, int * hist_classes){

    int i;
    int total_vals = 0;
    for(i=0; i<num_classes; i++){
        printf("Number of values in Class %d: %d\n", i, hist_classes[i]);
        total_vals += hist_classes[i];
    }
    printf("total values in histogram: %d\n", total_vals);
}

void group_data_bins(int local_data_len, int num_classes){

    int i, class_;
    for(i = 0; i < local_data_len; i++) 
    {
        class_ = find_bin(local_data[i], num_classes);  
        assert(("class index should not be negative", class_>=0));
        local_classes[class_] += 1;
    }

}

int find_bin(int local_data_val, int num_classes){

    int i;
    for(i = 0; i < num_classes; i++) 
    {
        if(local_data_val>=min_range_cls[i] && local_data_val<max_range_cls[i])
            return  i;
    }

    return -1;
}

int main(int argc, char *argv[])
{
    srand(time(NULL));

    int num_procs, proc_rank, name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    int global_data_len, num_classes, local_data_len;

    assert(("./Q2 <number of values> <number of classes>", argc == 3));

    global_data_len = atoi(argv[1]);
    num_classes = atoi(argv[2]);
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Get_processor_name(processor_name, &name_len); 

    // Data initialize
    gen_data(global_data_len);   
    set_classes(num_classes);

    if(proc_rank == 0){
        print_data(global_data_len, num_classes);
    }

    local_data_len = global_data_len / num_procs;  //data should be multiple of num_nodes

    MPI_Bcast(&num_classes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_data_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&local_data_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    local_data = (int *)calloc(local_data_len, sizeof(int));

    MPI_Scatter(data, local_data_len, MPI_INT, local_data, local_data_len,MPI_INT, 0, MPI_COMM_WORLD);

    classes = (int *)calloc(num_classes, sizeof(int));

    group_data_bins(local_data_len, num_classes);

    printf("Histogram for process %d on %s out of %d:\n", proc_rank, processor_name, num_procs);
    print_hist(num_classes, local_classes);

    MPI_Reduce(local_classes, classes, num_classes, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Finalize();

    if(proc_rank==0){
        printf("\nFinal Histogram:\n");
        print_hist(num_classes, classes);
    }

    return 0;

}