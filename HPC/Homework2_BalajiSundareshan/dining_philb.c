#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>

# define END_TIME 5000 //milliseconds

typedef struct {

    int position;
    int count;
    int fixed;
    int num_times_think;
    int num_times_eat;
    float current_wait_duration;
    int total_think_duration;
    int total_eat_duration;
    float total_wait_duration;
    sem_t *left_fork;
    sem_t *right_fork;

} philosopher_t;

void create_forks(sem_t *forks, int num_philosophers);
void start_threads(pthread_t *threads, sem_t *forks, int num_philosophers, int fixed);
void print_philosopher_stats(philosopher_t * philosopher);
void *start_activity_philosopher(void *arg);

void think(philosopher_t *philosopher, float max_think_time);
void eat(philosopher_t *philosopher, float max_eat_time);
void take_forks(philosopher_t *philosopher);
void place_forks(philosopher_t *philosopher);
int get_microseconds(int milliseconds);
int is_last_philosopher(philosopher_t *philosopher);
float get_random_number(int min, int max);

int main(int argc, char *argv[])
{
    srand(time(NULL));
    assert(("executable num_philosopher fixed_time_eat_flag", argc == 3));

    int num_philosophers = atoi(argv[1]);
    int fixed = atoi(argv[2]);

    printf("Number of Philosophers: %d\n", num_philosophers);
    printf("Expected Running Time: %d secs \n", END_TIME/1000);

    sem_t forks[num_philosophers];
    pthread_t threads[num_philosophers];

    create_forks(forks, num_philosophers);
    start_threads(threads, forks, num_philosophers, fixed); 
    
    pthread_exit(NULL);
    printf("End of Execution\n");
}

void create_forks(sem_t *forks, int num_philosophers)
{
  for(int i = 0; i < num_philosophers; i++) {
    sem_init(&forks[i], 0, 1);
  }
}

void start_threads(pthread_t *threads, sem_t *forks, int num_philosophers, int fixed)
{
  for(int i = 0; i < num_philosophers; i++) {
    philosopher_t *philosopher = malloc(sizeof(philosopher_t));
    
    philosopher->position = i;
    philosopher->count = num_philosophers;
    philosopher->fixed = fixed;
    philosopher->num_times_think = 0;
    philosopher->num_times_eat = 0;
    philosopher->current_wait_duration = 0.0;
    philosopher->total_think_duration = 0;
    philosopher->total_eat_duration = 0;
    philosopher->total_wait_duration = 0.0;
    philosopher->left_fork = &forks[i];
    philosopher->right_fork = &forks[(i + 1) % num_philosophers];
    
    pthread_create(&threads[i], NULL, start_activity_philosopher, (void *)philosopher);
  }
}

void *start_activity_philosopher(void *arg)
{
  philosopher_t *philosopher = (philosopher_t *)arg;
  time_t start_t, end_t;
  time(&start_t);
  while(1)
  {
    think(philosopher, 500);
    take_forks(philosopher);
    eat(philosopher, 500);
    place_forks(philosopher);
    time(&end_t);
    
    if(difftime(end_t, start_t)*1000>END_TIME){
        print_philosopher_stats(philosopher);
        break;
    }
  }
}

void print_philosopher_stats(philosopher_t * philosopher){

  printf("--------------------------------------------\n");
  printf("Philosopher %d stats\n",philosopher->position);
  printf("Philosopher number of times thought: %d \n",philosopher->num_times_think);
  printf("Philosopher number of plates eaten: %d \n",philosopher->num_times_eat);
  printf("Philosopher total eat duration: %d ms\n",philosopher->total_eat_duration);
  printf("Philosopher total think duration: %d ms\n",philosopher->total_think_duration);
  printf("Philosopher total wait duration: %f ms\n",philosopher->total_wait_duration);
  printf("--------------------------------------------\n");

}

void think(philosopher_t *philosopher, float max_think_time)
{

    int think_duration; 
    if (philosopher->fixed==1){
      think_duration = 500;
    }else{
      think_duration = get_random_number(100, max_think_time);
    }
    
    philosopher->total_think_duration += think_duration;

    printf("Philosopher %d started thinking for %d ms\n", philosopher->position, think_duration);

    // sleep
    float microsec = get_microseconds(think_duration);
    usleep(microsec);

    philosopher->num_times_think += 1;

    printf("Philosopher %d stopped thinking\n", philosopher->position);

}

int get_microseconds(int milliseconds){

    return milliseconds*1000;
}

void eat(philosopher_t *philosopher, float max_eat_time)
{
    int eat_time;
    if (philosopher->fixed==1){
      eat_time = 500;
    } else{
      eat_time = get_random_number(100, max_eat_time);
    }
    philosopher->total_eat_duration += eat_time;

    printf("Philosopher %d started eating for %d ms\n", philosopher->position, eat_time);
    // sleep
    float microsec = get_microseconds(eat_time);
    usleep(microsec);

    philosopher->num_times_eat += 1;

    printf("Philosopher %d stopped eating\n", philosopher->position);

}

void take_forks(philosopher_t *philosopher)
{ 
    time_t start_t, end_t;
    time(&start_t);
    if (is_last_philosopher(philosopher))
    {
        sem_wait(philosopher->left_fork);
        sem_wait(philosopher->right_fork);
    }
    else
    {
        sem_wait(philosopher->right_fork);
        sem_wait(philosopher->left_fork);
    }
    time(&end_t);
    philosopher->current_wait_duration = difftime(end_t, start_t)*1000; //milliseconds
    philosopher->total_wait_duration += philosopher->current_wait_duration;
    printf("Philosopher %d waited for forks for %f ms\n", philosopher->position, philosopher->current_wait_duration);
}

void place_forks(philosopher_t *philosopher)
{
    sem_post(philosopher->right_fork);
    sem_post(philosopher->left_fork);
}

int is_last_philosopher(philosopher_t *philosopher)
{
    return philosopher->position == philosopher->count-1;
}

float get_random_number(int min, int max)
{

    float result = (rand() % (max + 1));
    if (result < min) result = min;
    return result;

}
