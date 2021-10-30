#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>

int *numbers = NULL;
int largest_number;
int num_threads;
int total_numbers;
int div1, div2;
pthread_mutex_t sum_mutex;

void * find_numbers(void *arg);
void print_numbers();
int min(int a, int b);

typedef struct {

    int id;
    int start;
    int end;
    sem_t * sum_lock;
    
} thread_t;

int min(int a, int b){

    if(a<=b)
        return a;
    return b;
}

void * find_numbers(void *arg){


    thread_t *thread = (thread_t *)arg;
    printf("Thread ID: %d started\n", thread->id);

    int i;
    int local_sum = 0;
    for(i = thread->start; i < thread->end; i++){
        if(i<min(div1, div2)) continue;
        if(i % div1 == 0 || i % div2 == 0){
            numbers[i] = 1;
        }else{
            numbers[i] = 0;
        }
        if (numbers[i]==1)
            local_sum += 1;    
    }

    sem_wait(thread->sum_lock);
    total_numbers += local_sum;
    sem_post(thread->sum_lock);

    pthread_exit((void*) 0);

}

void print_numbers(){

    printf("List of divisible numbers:\n");
    int i;
    for(i=min(div1, div2); i<=largest_number; i++){
        if (numbers[i]==1){ 
            printf("%d, ",i);
        }
    }
}

int main(int argc, char *argv[])
{
    srand(time(NULL));
    assert(("./Q2 <largest_number> <num_threads> div1 div2", argc == 5));

    largest_number = atoi(argv[1]);
    num_threads = atoi(argv[2]);
    div1 = atoi(argv[3]);
    div2 = atoi(argv[4]);

    printf("Largest Number: %d\n", largest_number);
    printf("Number of Threads: %d\n", num_threads);
    printf("Divisor 1: %d\n", div1);
    printf("Divisor 2: %d\n", div2);
    int i;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    numbers = calloc(largest_number + 1, sizeof(int));
    pthread_t threads[num_threads];
    int thread_len = largest_number/num_threads;

    for(i=0; i<num_threads; i++){

        thread_t *thread = malloc(sizeof(thread_t));
        sem_t lock;

        thread->id = i;        
        thread->start = thread->id * thread_len;
        thread->end = thread->start + thread_len;

        if (thread->id == num_threads-1){
             
            thread->end = largest_number + 1;
        }

        sem_init(&lock, 0, 1);
        thread->sum_lock = &lock;

        pthread_create(&threads[i], NULL, find_numbers, (void *)thread);
    }

    for(i=0; i<num_threads; i++)
    {
        pthread_join(threads[i], NULL);
	}

    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec);
    time_taken += (end.tv_nsec - start.tv_nsec) / 1000000000.0;

    // print divisible numbers
    //print_numbers();
    printf("\nTotal number of numbers: %d\n", total_numbers);
    printf("Total time elapsed: %f seconds \n", time_taken);

    // free numbers
    free(numbers); 

    return 0;
}