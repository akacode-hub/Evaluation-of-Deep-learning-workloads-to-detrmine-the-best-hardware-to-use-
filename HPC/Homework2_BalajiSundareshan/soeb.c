#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <limits.h>

int primes[INT_MAX/2];
int largest_number;
int num_threads;

void *find_primes(void *arg){

    int local_len = largest_number/num_threads;
    int threadid = (long)arg;

    primes[0] = 0;
    primes[1] = 0;
    primes[2] = 1;
    int start = threadid * local_len;
    int end = start + local_len;

    if (threadid == num_threads-1){
        end = largest_number + 1;
    }

    int i;

    printf("Initialized thread ID: %d\n", threadid);
    // printf("local_len: %d\n", local_len);
    // printf("start: %d\n", start);
    // printf("end: %d\n", end);

    for(i=start; i<=end; i++){
        if(i<3)continue;
        if(i%2==0){
            primes[i] = 0;
        }else{
            primes[i] = 1;
        }
    }

    for(i=start; i<end; i++){
        if (primes[i]==1){
            for(int x=3; x*x<=i; x++){
                if (x % 2 != 0 && i % x == 0){
                    primes[i] = 0;
                    break;
                }
            }
        }
    }

    pthread_exit((void*) 0);

}

void print_primes(){

    int total_primes = 0;
    printf("List of prime numbers:\n");
    for(int i=0; i<=largest_number; i++){
        if (primes[i]==1){ 
            printf("%d, ",i);
            total_primes += 1;
        }
    }
    printf("\nTotal number of primes: %d\n", total_primes);
}

int main(int argc, char *argv[])
{
    srand(time(NULL));
    assert(("executable largest_number num_threads", argc == 3));

    largest_number = atoi(argv[1]);
    num_threads = atoi(argv[2]);

    printf("Largest Number: %d\n", largest_number);
    printf("Number of Threads: %d\n", num_threads);

    int i;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    pthread_t threads[num_threads];
    for(i=0; i<num_threads; i++){
        pthread_create(&threads[i], NULL, find_primes, (void *)i);
    }

    for(i=0; i<num_threads; i++)
    {
        pthread_join(threads[i], NULL);
	}

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec);
    time_taken += (end.tv_nsec - start.tv_nsec) / 1000000000.0;

    // print prime numbers
    print_primes();
    printf("Total time elapsed: %f seconds \n", time_taken);

    return 0;
}