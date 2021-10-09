#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>

int *primes = NULL;
void print_primes(int *primes, int largest_number);

void print_primes(int *primes, int largest_number){

    printf("List of prime numbers:\n");
    int i;
    for(i=0; i<=largest_number; i++){
        if (primes[i]==1){ 
            printf("%d, ",i);
        }
    }
}

int main(int argc, char *argv[])
{
    srand(time(NULL));
    assert(("./soe_omp <largest_number> <num_threads>", argc == 3));

    int largest_number = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    printf("Largest Number: %d\n", largest_number);
    printf("Number of Threads: %d\n", num_threads);

    int i;
    omp_set_num_threads(num_threads);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    primes = calloc(largest_number + 1, sizeof(int));
    primes[0] = 0;
    primes[1] = 0;
    primes[2] = 1;

    #pragma omp parallel for
    for(i=3; i<largest_number + 1; i++){
        int tid = omp_get_thread_num();
        if(i%2==0){
            primes[i] = 0;
        }else{
            primes[i] = 1;
        }
    }
    
    #pragma omp parallel for schedule(dynamic, 100)
    for(i=3; i<largest_number + 1; i++){
        if(primes[i] == 1){
            int x;
            for(x=3; x*x<=i; x++){
                if(x % 2 != 0 && i % x == 0){
                    primes[i] = 0;
                    break;
                }
            }
        }
    }

    // print prime numbers
    int total_primes = 0;
    #pragma omp parallel for reduction(+:total_primes)
    for(i=0; i<largest_number + 1; i++){
        if (primes[i]==1){ 
            total_primes += 1;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec);
    time_taken += (end.tv_nsec - start.tv_nsec) / 1000000000.0;

    //print_primes(primes, largest_number);
    printf("\nTotal number of primes: %d\n", total_primes);
    printf("Total time elapsed: %f seconds \n", time_taken);

    return 0;
}