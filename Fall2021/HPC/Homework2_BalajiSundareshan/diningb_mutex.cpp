# include <cstdlib>
# include <pthread.h>
# include <stdio.h>
# include <unistd.h>
# include <cassert>
# include <string>

using namespace std;

struct philosopher_t {
    int position;
    int count;
    int current_think_duration = 0;
    int current_eat_duration = 0;
    int current_wait_duration = 0;
    int total_think_duration = 0;
    int total_eat_duration = 0;
    int total_wait_duration = 0;
    pthread_mutex_t left_fork;
    pthread_mutex_t right_fork;
};

int get_random_number(int min, int max);
void start_threads(pthread_t *threads, pthread_mutex_t *forks, int num_philosophers);
void think(philosopher_t *philosopher, int max_think_time);
void eat(philosopher_t *philosopher, int max_eat_time);
void *start_activity_philosopher(void *arg);
int is_last_philosopher(philosopher_t *philosopher);
void create_forks(pthread_mutex_t *forks, int num_philosophers);
bool take_forks(philosopher_t *philosopher, pthread_mutex_t left_fork, pthread_mutex_t right_fork);

void start_threads(pthread_t *threads, pthread_mutex_t *forks, int num_philosophers)
{

    for(int i = 0; i < num_philosophers; i++) {

        philosopher_t *philosopher = new philosopher_t;
        philosopher->position = i;
        philosopher->count = num_philosophers;
        philosopher->left_fork = forks[i];
        philosopher->right_fork = forks[i + 1 % num_philosophers];
        pthread_create(&threads[i], NULL, start_activity_philosopher, (void *)philosopher);
  }
}

void think(philosopher_t *philosopher, int max_think_time)
{

    philosopher->current_think_duration = get_random_number(1, max_think_time);

    philosopher->total_think_duration += philosopher->current_think_duration;

    printf("Philosopher %d started thinking for %d seconds\n", philosopher->position, philosopher->current_think_duration);
    // sleep
    sleep(philosopher->current_think_duration);

    printf("Philosopher %d stopped thinking\n", philosopher->position);

}

void eat(philosopher_t *philosopher, int max_eat_time)
{

    philosopher->current_eat_duration = get_random_number(1, max_eat_time);
    philosopher->total_eat_duration += philosopher->current_eat_duration;

    printf("Philosopher %d started eating for %d seconds\n", philosopher->position, philosopher->current_eat_duration);
    // sleep
    sleep(philosopher->current_eat_duration);

    printf("Philosopher %d stopped eating\n", philosopher->position);

}

bool take_forks(philosopher_t *philosopher, pthread_mutex_t left_fork, pthread_mutex_t right_fork)
{

  while(1){
    if (pthread_mutex_trylock (&left_fork) == 0){
    
      //printf("Philosopher %d got the left fork\n ", philosopher->position);

      if (pthread_mutex_trylock (&right_fork) == 0){

        printf("Philosopher %d got both forks\n ", philosopher->position);

        return true;

      } else {

        pthread_mutex_unlock (&left_fork);

      }
    }
  } 
  return false;
}

// bool take_forks(philosopher_t *philosopher, pthread_mutex_t left_fork, pthread_mutex_t right_fork)
// {

//   int left = pthread_mutex_trylock (&left_fork);

//   if (left==0){

//       int right = pthread_mutex_trylock (&right_fork);

//       if (right==0){
//   }
  
// }

void place_forks(philosopher_t *philosopher)
{
    pthread_mutex_unlock(&philosopher->right_fork);
    pthread_mutex_unlock(&philosopher->left_fork);
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
    bool status;

    while(true)
    {
        think(philosopher, 2);
        if (is_last_philosopher(philosopher))
        {
          take_forks(philosopher, philosopher->right_fork, philosopher->left_fork);  
        } else {
          take_forks(philosopher, philosopher->left_fork, philosopher->right_fork);  
        }
        
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