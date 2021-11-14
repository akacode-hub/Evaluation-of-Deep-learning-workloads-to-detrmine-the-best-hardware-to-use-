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
int * data_procs;
int * classes;
int * classes_procs;
float * min_range_cls;
float * max_range_cls;

void gen_data(int num_values)
{   

    unsigned int seed = (unsigned) time(NULL);
	srand(seed);

    data = (int *)calloc(num_values, sizeof(int));
    int i;

    for(i=0; i<num_values; i++){
        data[i] = rand()%MAX_NUM + MIN_NUM;
    }
}

void set_classes(int num_classes){

    classes_procs = (int *)calloc(num_classes, sizeof(int));
    min_range_cls = (float *)calloc(num_classes, sizeof(float));
    max_range_cls = (float *)calloc(num_classes, sizeof(float));
    
    int i;
    
    float range = MAX_NUM - MIN_NUM;
    float interval = (float) range / (float) num_classes;
    interval = ceil(interval);

    for(i=0; i<num_classes; i++){

        min_range_cls[i] = i*interval + MIN_NUM;
        max_range_cls[i] = (i+1)*interval + MIN_NUM;
        classes_procs[i] = 0;
    }

}

void print_data(int num_values, int num_classes){

    int i;

    printf("Input data: ");
    for(i=0; i<num_values; i++){
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
    printf("total vals: %d\n", total_vals);
}

void group_data_bins(int num_values_proc, int num_classes){

    int i, class_;
    for(i = 0; i < num_values_proc; i++) 
    {
        class_ = find_bin(data_procs[i], num_classes);  
        classes_procs[class_] += 1;
    }

}

int find_bin(int local_data_val, int num_classes){

    int i;
    for(i = 0; i < num_classes; i++) 
    {
        if(local_data_val>=min_range_cls[i] && local_data_val<max_range_cls[i])
            return  i;
    }

}

int main(int argc, char *argv[])
{
    srand(time(NULL));

    int num_procs, proc_rank, name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    int num_values, num_classes, num_values_proc;

    assert(("./Q2 <number of values> <number of classes>", argc == 3));

    num_values = atoi(argv[1]);
    num_classes = atoi(argv[2]);
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Get_processor_name(processor_name, &name_len); 

    // Data initialize
    if(proc_rank == 0){
        gen_data(num_values);   
        set_classes(num_classes);
        print_data(num_values, num_classes);
    }

    num_values_proc = num_values / num_procs;

    MPI_Bcast(&num_classes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_values, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_values_proc, 1, MPI_INT, 0, MPI_COMM_WORLD);

    data_procs = (int *)calloc(num_values, sizeof(int));
    MPI_Scatter(data, num_values_proc, MPI_INT, data_procs, num_values_proc,MPI_INT, 0, MPI_COMM_WORLD);

    classes = (int *)calloc(num_classes, sizeof(int));

    group_data_bins(num_values_proc, num_classes);

    MPI_Reduce(classes_procs, classes, num_classes, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Finalize();

    if(proc_rank==0){
        print_hist(num_classes, classes);
    }

    return 0;

}