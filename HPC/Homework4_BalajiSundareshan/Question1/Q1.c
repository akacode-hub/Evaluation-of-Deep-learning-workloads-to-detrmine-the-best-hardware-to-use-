#include <stdbool.h>
#include <mpi.h>  
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

double PI25DT = 3.141592653589793238462643;         /* 25-digit-PI*/

bool check_pt_in_circle(double x, double y){
    
    if(x*x + y*y <= 1.0){return true;}
    return false;
}

long count_toss_in_circle(long num_tosses_process, int process_rank)
{
    unsigned int seed = (unsigned) time(NULL);
    seed = seed + process_rank;
	srand(seed);

    long toss;
    double x, y;
    long num_toss_in = 0;
    bool dart_in_circle;

    for(toss = 0; toss < num_tosses_process; toss++){

        x = rand_r(&seed)/(double)RAND_MAX;
	    y = rand_r(&seed)/(double)RAND_MAX;

        dart_in_circle = check_pt_in_circle(x, y); 
        if(dart_in_circle){
            num_toss_in += 1;
        }
    }
    return num_toss_in;
}

int main(int argc, char *argv[])
{
    srand(time(NULL));

    int num_procs, proc_rank, name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    long num_darts, num_darts_procs, num_darts_procs_circle, num_darts_circle;
    double start_time, end_time, time_elapsed_procs, total_time_elapsed;
    double calculated_pi;

    assert(("./Q1 <number of darts> <number of nodes>", argc == 3));

    num_darts = atoi(argv[1]);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Get_processor_name(processor_name, &name_len); 

    MPI_Bcast(&num_darts, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    num_darts_procs = num_darts/num_procs;

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    num_darts_procs_circle = count_toss_in_circle(num_darts_procs, proc_rank);
    end_time = MPI_Wtime();
    time_elapsed_procs = end_time - start_time;
    
    // reduction in time
    MPI_Reduce(&time_elapsed_procs, &total_time_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); 
    
    // reduction in tosses
    MPI_Reduce(&num_darts_procs_circle, &num_darts_circle, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    printf("Number of darts for process %d on %s out of %d: %ld\n", proc_rank, processor_name, num_procs, num_darts_procs_circle);

    MPI_Finalize(); 

    if (proc_rank == 0) {
        calculated_pi = (4*num_darts_circle)/((double) num_darts);
        double error = fabs(calculated_pi - PI25DT);
        printf("Elapsed time = %f seconds \n", total_time_elapsed);
        printf("Pi is approximately %.16f, Error is %.16f\n", calculated_pi, error);
    }

    return 0;
}