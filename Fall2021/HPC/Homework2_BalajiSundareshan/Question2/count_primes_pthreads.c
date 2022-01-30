#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>

int *primes = NULL;
int largest_number;
int num_threads;
int total_num_primes;
pthread_mutex_t sum_mutex;

void *find_primes(void *arg);
void print_primes();

void *find_primes(void *arg){

    int local_len = largest_number/num_threads;
    int threadid = (long)arg;

    int start = threadid * local_len;
    int end = start + local_len;

    if (threadid == num_threads-1){
        end = largest_number + 1;
    }

    int i;

    for(i=start; i<=end; i++){
        if(i<3)continue;
        if(i%2==0){
            primes[i] = 0;
        }else{
            primes[i] = 1;
        }
    }

    double local_sum = 0;
    for(i=start; i<end; i++){
        if (primes[i]==1){
            int x;
            for(x=3; x*x<=i; x++){
                if (x % 2 != 0 && i % x == 0){
                    primes[i] = 0;
                    break;
                }
            }
        }
        if (primes[i]==1)
            local_sum += 1;
    }

    pthread_mutex_lock (&sum_mutex);
    total_num_primes += local_sum;
    pthread_mutex_unlock (&sum_mutex);

    pthread_exit((void*) 0);

}

void print_primes(){

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
    assert(("./count_primes_pthreads <largest_number> <num_threads>", argc == 3));

    largest_number = atoi(argv[1]);
    num_threads = atoi(argv[2]);

    printf("Largest Number: %d\n", largest_number);
    printf("Number of Threads: %d\n", num_threads);

    int i;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    primes = calloc(largest_number + 1, sizeof(int));
    pthread_t threads[num_threads];

    primes[0] = 0;
    primes[1] = 0;
    primes[2] = 1;

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
    printf("\nTotal number of primes: %d\n", total_num_primes);
    printf("Total time elapsed: %f seconds \n", time_taken);

    // free primes
    free(primes); 

    return 0;
}