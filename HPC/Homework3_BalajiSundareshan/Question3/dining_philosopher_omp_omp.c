#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <signal.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>

# define MAX_ITER 50

omp_lock_t sum_lock;
omp_lock_t print_lock;
float global_eat_duration = 0;
float global_think_duration = 0;
float global_wait_duration = 0;
int num_iter = 0;

typedef struct {

    int position;
    int count;
    int min_dur;
    int max_dur;
    int num_times_think;
    int num_times_eat;
    int total_think_duration;
    int total_eat_duration;
    float total_wait_duration;
    omp_lock_t *left_fork;
    omp_lock_t *right_fork;

} philosopher_t;

void create_forks(omp_lock_t *forks, int num_forks);
void start_activity_philosopher(philosopher_t * philosopher);
void think(philosopher_t *philosopher);
void take_forks(philosopher_t *philosopher);
void eat(philosopher_t *philosopher);
void place_forks(philosopher_t *philosopher);
int is_last_philosopher(philosopher_t *philosopher);
float get_random_number(int min, int max);
int get_microseconds(int milliseconds);
void print_philosopher_stats(philosopher_t * philosopher);

int main(int argc, char *argv[])
{
    srand(time(NULL));
    assert(("./dining_philosopher_omp <num_philosopher> <min_dur_microsec> <max_dur_microsec>", argc == 4));
    
    int num_philosophers = atoi(argv[1]);
    int min_dur = atoi(argv[2]);
    int max_dur = atoi(argv[3]);
    int num_forks = num_philosophers;

    assert(("max_dur should be >= min_dur", max_dur>=min_dur));

    printf("...................................\n");
    printf("Number of Philosophers: %d\n", num_philosophers);
    printf("Number of Forks: %d\n", num_forks);
    printf("Maximum iteration: %d secs \n", MAX_ITER);
    printf("Minimum Duration to Think/Eat: %d msecs \n", min_dur);
    printf("Maximum Duration to Think/Eat: %d msecs \n", max_dur);
    printf("...................................\n\n");

    struct timespec gstart, gend;
    clock_gettime(CLOCK_MONOTONIC, &gstart);

    omp_lock_t forks[num_forks];
    create_forks(forks, num_forks);

    #pragma omp parallel num_threads(num_philosophers)
    {
        int thread_num = omp_get_thread_num();
		printf("Thread ID: %d started!\n",thread_num);

        philosopher_t *philosopher = malloc(sizeof(philosopher_t));
        philosopher->position = thread_num;
        philosopher->count = num_philosophers;
        philosopher->num_times_think = 0;
        philosopher->num_times_eat = 0;
        philosopher->total_think_duration = 0;
        philosopher->total_eat_duration = 0;
        philosopher->total_wait_duration = 0;
        philosopher->min_dur = min_dur;
        philosopher->max_dur = max_dur;
        philosopher->left_fork = &forks[thread_num];
        philosopher->right_fork = &forks[(thread_num + 1) % num_philosophers];
        
        // Wait for all threads to start
        #pragma omp barrier
        start_activity_philosopher(philosopher);

    }

    clock_gettime(CLOCK_MONOTONIC, &gend);
    float time_elapsed = (gend.tv_sec - gstart.tv_sec);
    time_elapsed += (gend.tv_nsec - gstart.tv_nsec) / 1000000000.0;

    int i;
    for (i = 0; i < num_philosophers; i++)
	    omp_destroy_lock(&forks[i]);

    printf("\nTotal eat duration: %f\n", global_eat_duration);
    printf("Total wait duration: %f\n", global_wait_duration);
    printf("Total think duration: %f\n", global_think_duration);
    printf("Total time elapsed: %f\n", time_elapsed);
    printf("End of Execution\n");

    return 0;
}

void start_activity_philosopher(philosopher_t * philosopher)
{

  time_t gstart, gend;
  time(&gstart);

  while(1)
  {
    think(philosopher);
    take_forks(philosopher);
    eat(philosopher);
    place_forks(philosopher);
    time(&gend);
    
    omp_set_lock (&sum_lock);
    num_iter += 1;
    omp_unset_lock (&sum_lock);

    if(num_iter>MAX_ITER){
        sleep(1);
        print_philosopher_stats(philosopher);
        break;
    }
  }
}

void create_forks(omp_lock_t *forks, int num_forks)
{ 
    int i;
    for(i = 0; i < num_forks; i++) {
       omp_init_lock(&forks[i]);
    }
}

void think(philosopher_t *philosopher)
{

    int think_duration; 
    think_duration = get_random_number(philosopher->min_dur, philosopher->max_dur);
    philosopher->total_think_duration += think_duration;
    printf("Philosopher %d started thinking for %d ms\n", philosopher->position, think_duration);

    // sleep
    float microsec = get_microseconds(think_duration);
    usleep(microsec);

    philosopher->num_times_think += 1;

    printf("Philosopher %d stopped thinking\n", philosopher->position);

}

void take_forks(philosopher_t *philosopher)
{ 
    struct timespec wstart, wend;
    clock_gettime(CLOCK_MONOTONIC, &wstart);

    if (is_last_philosopher(philosopher))
    {
        omp_set_lock(philosopher->right_fork);
        omp_set_lock(philosopher->left_fork);
    }
    else
    {
        omp_set_lock(philosopher->left_fork);
        omp_set_lock(philosopher->right_fork);
    }

    clock_gettime(CLOCK_MONOTONIC, &wend);
    float wait_time = (wend.tv_sec - wstart.tv_sec);
    wait_time += (wend.tv_nsec - wstart.tv_nsec) / 1000000000.0;
    philosopher->total_wait_duration += wait_time*1000; //ms
}

void eat(philosopher_t *philosopher)
{
    int eat_time;
    eat_time = get_random_number(philosopher->min_dur, philosopher->max_dur);
    philosopher->total_eat_duration += eat_time;

    printf("Philosopher %d started eating for %d ms\n", philosopher->position, eat_time);
    // sleep
    float microsec = get_microseconds(eat_time);
    usleep(microsec);

    philosopher->num_times_eat += 1;

    printf("Philosopher %d stopped eating\n", philosopher->position);

}

void place_forks(philosopher_t *philosopher)
{   
    omp_unset_lock(philosopher->right_fork);
    omp_unset_lock(philosopher->left_fork);
}

int is_last_philosopher(philosopher_t *philosopher)
{
    return philosopher->position == philosopher->count-1;
}

float get_random_number(int min, int max)
{

    if (min==max){
      return min;
    }

    float result = (rand() % (max + 1));
    if (result < min) result = min;
    return result;

}

int get_microseconds(int milliseconds){

    return milliseconds*1000;
}

void print_philosopher_stats(philosopher_t * philosopher){

  omp_set_lock (&print_lock);
  printf("--------------------------------------------\n");
  printf("Philosopher %d stats\n",philosopher->position);
  printf("Philosopher %d number of times thought: %d \n", philosopher->position, philosopher->num_times_think);
  printf("Philosopher %d number of plates eaten: %d \n", philosopher->position, philosopher->num_times_eat);
  printf("Philosopher %d total eat duration: %d ms\n", philosopher->position, philosopher->total_eat_duration);
  printf("Philosopher %d total wait duration: %f ms\n", philosopher->position, philosopher->total_wait_duration);
  printf("Philosopher %d total think duration: %d ms\n", philosopher->position, philosopher->total_think_duration);
  printf("--------------------------------------------\n");

  global_eat_duration += philosopher->total_eat_duration;
  global_wait_duration += philosopher->total_wait_duration;
  global_think_duration += philosopher->total_think_duration;

  omp_unset_lock (&print_lock);

}