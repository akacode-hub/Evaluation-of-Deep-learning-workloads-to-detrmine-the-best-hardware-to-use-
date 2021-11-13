#include <stdbool.h>
#include <mpi.h>  
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define MIN_NUM 1
#define MAX_NUM 1000
int * inp_arr;

void gen_inp_array(int num_values)
{   
    inp_arr = (int *)calloc(num, sizeof(int));
    int i;

    for(i=0; i<=num; i++){
        inp_arr[i] = rand()%MAX_NUM + 1;
    }

}


int main(int argc, char *argv[])
{
    srand(time(NULL));

    int num_procs, proc_rank, name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    assert(("./Q2 <number of values> <number of classes>", argc == 3));

    num_values = atoi(argv[1]);
    num_classes = atoi(argv[2]);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Get_processor_name(processor_name, &name_len); 



    return 0;
}