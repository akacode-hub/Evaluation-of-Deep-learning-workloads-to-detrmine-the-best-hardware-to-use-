# include <cstdlib>
# include <pthread.h>
# include <stdio.h>
# include <unistd.h>
# include <cassert>

struct philosopher_t {
    int position;
    int count;
    float total_think_duration;
    float total_eat_duration;
    float total_wait_duration;
    pthread_mutex_t *left_fork;
    pthread_mutex_t *right_fork;
};

int get_random_number(int min, int max);
void start_threads(pthread_t *threads, pthread_mutex_t *forks, int num_philosophers);
void think(philosopher_t *philosopher, int max_think_time);
void eat(philosopher_t *philosopher, int max_eat_time);
void *start_activity_philosopher(void *arg);
int is_last_philosopher(philosopher_t *philosopher);
void create_forks(pthread_mutex_t *forks, int num_philosophers);

void start_threads(pthread_t *threads, pthread_mutex_t *forks, int num_philosophers)
{

    for(int i = 0; i < num_philosophers; i++) {

        philosopher_t *philosopher = new philosopher_t;
        printf("phil init\n");
        philosopher->position = i;
        printf("phil init pos\n");
        philosopher->count = num_philosophers;
        philosopher->total_think_duration = 0.0;
        philosopher->total_eat_duration = 0.0;
        philosopher->total_wait_duration = 0.0;
        philosopher->left_fork = &forks[i];
        philosopher->right_fork = &forks[(i + 1) % num_philosophers];
        pthread_create(&threads[i], NULL, start_activity_philosopher, (void *)philosopher);
  }
}

void think(philosopher_t *philosopher, int max_think_time)
{

    int philosopher_id = philosopher->position + 1;
    int think_time = get_random_number(1, max_think_time);
    philosopher->total_think_duration += think_time;

    printf("Philosopher %d started thinking\n", philosopher_id);
    // sleep
    sleep(think_time);

    printf("Philosopher %d stopped thinking after %f\n", (philosopher_id, think_time));

}

void eat(philosopher_t *philosopher, int max_eat_time)
{

    int philosopher_id = philosopher->position + 1;
    int eat_time = get_random_number(1, max_eat_time);
    philosopher->total_eat_duration += eat_time;

    printf("Philosopher %d started eating\n", philosopher_id);
    // sleep
    sleep(eat_time);

    printf("Philosopher %d stopped eating after %f\n", (philosopher_id, eat_time));

}

void take_forks(philosopher_t *philosopher)
{
  if (is_last_philosopher(philosopher))
  {
    printf("right Philosopher %d lock\n", philosopher->position + 1);
    pthread_mutex_lock (philosopher->right_fork);
    pthread_mutex_lock (philosopher->left_fork);
    printf("right Philosopher %d lock END\n", philosopher->position + 1);
  }
  else
  {
    printf("left Philosopher %d lock\n", philosopher->position + 1);
    pthread_mutex_lock(philosopher->left_fork);
    pthread_mutex_lock(philosopher->right_fork);
  }
}

void place_forks(philosopher_t *philosopher)
{
    printf("Philosopher %d unlock\n", philosopher->position + 1);
    pthread_mutex_unlock(philosopher->right_fork);
    pthread_mutex_unlock(philosopher->left_fork);
}

int is_last_philosopher(philosopher_t *philosopher)
{

    return philosopher->position == philosopher->count - 1;
    
}
    

int get_random_number(int min, int max)
{
  
    int result = (rand() % (max + 1));
    if (result < min) result = min;
    return result;

}

void *start_activity_philosopher(void *arg)
{
    philosopher_t *philosopher = (philosopher_t *) arg;

    while(true)
    {
        think(philosopher, 2);
        take_forks(philosopher);
        eat(philosopher, 2);
        place_forks(philosopher);
    }

}

void create_forks(pthread_mutex_t *forks, int num_philosophers)
{
  for(int i = 0; i < num_philosophers; i++) {
    pthread_mutex_t forks[i] = PTHREAD_MUTEX_INITIALIZER;

  }
}

int main(int argc, char *argv[])
{

    srand(time(NULL));

    assert(("provide number of philosophers !!", argc == 2));

    int num_philosophers = atoi(argv[1]);

    printf("Number of Philosophers: %d\n", num_philosophers);

    pthread_mutex_t forks[num_philosophers];
    pthread_t threads[num_philosophers];

    create_forks(forks, num_philosophers);
    printf("start threads:\n");
    start_threads(threads, forks, num_philosophers);

    pthread_exit(NULL);

}