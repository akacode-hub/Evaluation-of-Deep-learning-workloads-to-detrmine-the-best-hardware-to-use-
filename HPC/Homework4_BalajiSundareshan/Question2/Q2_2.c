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
int * classes;
int * local_classes;
int * class_ids;
float * min_range_cls;
float * max_range_cls;

void gen_data(int data_len)
{

    unsigned int seed = (unsigned) time(NULL);
	srand(seed);

    data = (int *)calloc(data_len, sizeof(int));

    int i;
    for(i=0; i<data_len; i++){
        data[i] = rand() % MAX_NUM + MIN_NUM;
    }

}

void set_classes(int num_classes){

    local_classes = (int *)calloc(num_classes, sizeof(int));
    class_ids = (int *)calloc(num_classes, sizeof(int));
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
        class_ids[i] = i;
    }

}

void print_data(int data_len, int num_classes){

    int i;

    printf("Input data: ");
    for(i=0; i<data_len; i++){
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

void group_bins(int data_len, int class_id){

    int i, class_;
    for(i = 0; i < data_len; i++) 
    {
        if(data[i] >= min_range_cls[class_id] && data[i] < max_range_cls[class_id]){
            local_classes[class_] += 1;
        }
    }
}

int main(int argc, char *argv[])
{
    srand(time(NULL));

    int num_procs, proc_rank, name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    int data_len, num_classes;

    assert(("./Q2 <number of values> <number of classes>", argc == 3));

    data_len = atoi(argv[1]);
    num_classes = atoi(argv[2]);
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Get_processor_name(processor_name, &name_len); 

    // Data initialize
    gen_data(data_len);   
    set_classes(num_classes);
    
    if(proc_rank == 0){
        print_data(data_len, num_classes);
    }

    MPI_Bcast(&num_classes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&data_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    classes = (int *)calloc(num_classes, sizeof(int));

    group_bins(data_len, class_id)

    MPI_Reduce(local_classes, classes, num_classes, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Finalize();

    if(proc_rank==0){
        print_hist(num_classes, classes);
    }

    return 0;

}