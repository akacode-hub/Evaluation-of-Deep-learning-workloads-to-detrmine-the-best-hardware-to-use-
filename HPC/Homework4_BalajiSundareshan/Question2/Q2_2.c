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
int * local_data_lens;
int * class_dist_proc_count;
int * data_stride;
int * displs;
int num_procs, global_data_len;
int num_classes;
int proc_rank, name_len;
char processor_name[MPI_MAX_PROCESSOR_NAME];

int * classes;
float * min_range_cls;
float * max_range_cls;

int * result_len;
int * global_result;

void gen_data(int global_data_len)
{

    data = (int *)calloc(global_data_len, sizeof(int));

    int i;
    for(i=0; i<global_data_len; i++){
        data[i] = rand() % MAX_NUM + MIN_NUM;
    }

}

void set_classes(int num_classes){

    min_range_cls = (float *)calloc(num_classes, sizeof(float));
    max_range_cls = (float *)calloc(num_classes, sizeof(float));
    
    int i;
    
    float range = MAX_NUM - MIN_NUM;
    float interval = (float) range / (float) num_classes;
    interval = ceil(interval);

    for(i=0; i<num_classes; i++){

        min_range_cls[i] = i*interval + MIN_NUM;
        max_range_cls[i] = (i+1)*interval + MIN_NUM;
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

void group_class_bins(int data_len, int proc_rank, int class_dist_proc[num_procs][num_classes], int result[num_classes * global_data_len])
{

    int i, len;
    int * class_ids = class_dist_proc[proc_rank];
    for(i = 0; i < data_len; i++) 
    {   
        len = class_dist_proc_count[proc_rank];
        find_bin(data[i], class_ids, len, result);    
    }

}

int find_bin(int val, int * class_ids, int id_len, int result[num_classes * global_data_len])
{

    int i, cls_id, idx;
    for(i = 0; i < id_len; i++) 
    {   
        cls_id = class_ids[i];
        if(val >= min_range_cls[cls_id] && val < max_range_cls[cls_id]){

            idx = result_len[cls_id];
            result[cls_id * global_data_len + idx] = val;
            result_len[cls_id] += 1;
            return 1;   
        }       
    }
}

void dist_data(int global_data_len, int num_classes, int num_proc){

    local_data_lens = (int *)calloc(num_proc, sizeof(int));
    int data_bin_len = global_data_len / num_classes;

    int i;
    for(i=0; i<num_classes; i++){
        local_data_lens[i%num_proc] += data_bin_len;
    }

    displs = (int *)calloc(num_proc, sizeof(int));
    
    displs[i] = 0;
    for (i=1; i<num_proc; i++) { 
        displs[i] = displs[i-1] + local_data_lens[i-1]; 
    } 
}

void class_dist(int num_classes, int num_procs, int class_dist_proc[num_procs][num_classes]){
    
    int i, id;
    class_dist_proc_count = (int *)calloc(num_procs, sizeof(int));

    for(i=0; i<num_classes; i++){
        id = class_dist_proc_count[i%num_procs];
        class_dist_proc[i%num_procs][id] = i;
        class_dist_proc_count[i%num_procs] += 1;
    }
}

void print_class_dist(int class_dist_proc[num_procs][num_classes]){

    int i, j, len;
    for(i=0; i<num_procs; i++){
        len = class_dist_proc_count[i];
        printf("proc id %d: ", i);
        for(j=0; j<len; j++){
            printf("%d, ",class_dist_proc[i][j]);
        }
    }
}

void print_scatter_params(int num_proc){

    int i, j;
    printf("Data len for each proc:\n");
    for(i=0; i<num_proc; i++){
        printf("proc %d: %d\n", i, local_data_lens[i]);
    }
    
    printf("Displs for each proc:\n");
    for(i=0; i<num_proc; i++){
        printf("proc %d: %d\n", i, displs[i]);
    }
}

void print_data_buffer(int proc_rank){

    int i;
    for (i = 0; i < local_data_lens[proc_rank]; i++) {
        printf("%d, ", local_data[i]);
    }
    printf("\n");
}

void print_results(int result[num_classes * global_data_len])
{
    int i, j, len;
    for(i=0; i<num_classes; i++){
        len = classes[i];
        printf("cls id: %d ", i);
        for(j=0; j<len; j++){
            printf("%d ", result[i * global_data_len + j]);
        }
        printf("\n");
    }
}

void initialize_result(int result[num_classes * global_data_len]){

    int i;
    for(i=0; i<num_classes * global_data_len; i++){
        result[i] = 0;
    }
}

int main(int argc, char *argv[])
{

    assert(("./Q2_1 <number of values> <number of classes>", argc == 3));

    global_data_len = atoi(argv[1]);
    num_classes = atoi(argv[2]);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Get_processor_name(processor_name, &name_len); 

    // Data initialize
    gen_data(global_data_len);   
    set_classes(num_classes);

    MPI_Bcast(&num_classes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_data_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int class_dict_proc[num_procs][num_classes];
    int result[num_classes * global_data_len];
    initialize_result(result);

    dist_data(global_data_len, num_classes, num_procs);

    if(proc_rank == 0){
        print_data(global_data_len, num_classes);
        print_scatter_params(num_procs);
    }

    local_data = (int *)calloc(local_data_lens[proc_rank], sizeof(int));
    classes = (int *)calloc(num_classes, sizeof(int));
    result_len = (int *)calloc(num_classes, sizeof(int));
    global_result = (int *)calloc(num_classes * global_data_len, sizeof(int));

    MPI_Bcast(data, global_data_len, MPI_INT, 0, MPI_COMM_WORLD);
    printf("processor rank: %d len: %d\n", proc_rank, local_data_lens[proc_rank]);
    //print_data_buffer(proc_rank);

    class_dist(num_classes, num_procs, class_dict_proc);
    print_class_dist(class_dict_proc);

    group_class_bins(global_data_len, proc_rank, class_dict_proc, result);
    printf("Histogram for process %d on %s out of %d:\n", proc_rank, processor_name, num_procs);
    
    MPI_Reduce(result_len, classes, num_classes, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&result, global_result, num_classes*global_data_len, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Finalize();

    if(proc_rank==0){
        printf("\nFinal Histogram:\n");
        print_hist(num_classes, classes);
        print_results(global_result);
    }

    return 0;
}