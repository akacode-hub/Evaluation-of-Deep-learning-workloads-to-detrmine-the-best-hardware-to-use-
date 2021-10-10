#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>

# define END_TIME 2000 //milliseconds
pthread_mutex_t print_mutex;
int thread_priority_val = 50;

typedef struct {

    int position;
    int count;
    int extra_fork;
    int fork_status[3];
    int min_dur;
    int max_dur;
    int num_times_think;
    int num_times_eat;
    int total_think_duration;
    int total_eat_duration;
    sem_t *left_fork;
    sem_t *right_fork;
    sem_t *center_fork;
    
} philosopher_t;

void create_forks(sem_t *forks, int num_forks);
void start_threads(sem_t *forks, int num_philosophers, \
                    int min_dur, int max_dur, int prio_phil_id, \
                    int num_forks);
void print_philosopher_stats(philosopher_t * philosopher);
void *start_activity_philosopher(void *arg);
void pick_quick_fork(philosopher_t *philosopher, sem_t *right_fork, int last);
void think(philosopher_t *philosopher);
void eat(philosopher_t *philosopher);
void take_forks(philosopher_t *philosopher);
void place_forks(philosopher_t *philosopher);
int get_microseconds(int milliseconds);
int is_last_philosopher(philosopher_t *philosopher);
float get_random_number(int min, int max);

int main(int argc, char *argv[])
{
    srand(time(NULL));
    assert(("./dining_philosopher <num_philosopher> <min_dur_microsec> <max_dur_microsec> <priority_philosopher_id> <extra_fork>", argc == 6));
    
    int num_philosophers = atoi(argv[1]);
    int min_dur = atoi(argv[2]);
    int max_dur = atoi(argv[3]);
    int prio_phil_id = atoi(argv[4]); // -1 to disable
    int extra_fork = atoi(argv[5]); // +1/-1
    int num_forks = num_philosophers;
    num_forks += extra_fork;

    assert(("max_dur should be >= min_dur", max_dur>=min_dur));
    assert(("extra_fork should be equal to 1", extra_fork<=1));

    printf("...................................\n");
    printf("Number of Philosophers: %d\n", num_philosophers);
    printf("Number of Forks: %d\n", num_forks);
    printf("Expected Running Time: %d secs \n", END_TIME/1000);
    printf("Minimum Duration to Think/Eat: %d msecs \n", min_dur);
    printf("Maximum Duration to Think/Eat: %d msecs \n", max_dur);
    printf("...................................\n\n");
    
    sem_t forks[num_forks];
    pthread_t threads[num_philosophers];

    create_forks(forks, num_forks);
    start_threads(forks, num_philosophers, min_dur, max_dur, prio_phil_id, num_forks); 
    
    pthread_exit(NULL);
    printf("End of Execution\n");
}

void create_forks(sem_t *forks, int num_forks)
{ 
  int i;
  for(i = 0; i < num_forks; i++) {
    sem_init(&forks[i], 0, 1);
  }
}

void start_threads(sem_t *forks, int num_philosophers, int min_dur, int max_dur, int prio_phil_id, int num_forks)
{
  int i;
  pthread_t threads[num_philosophers];
  for(i = 0; i < num_philosophers; i++) {
    philosopher_t *philosopher = malloc(sizeof(philosopher_t));
    
    philosopher->position = i;
    philosopher->count = num_philosophers;
    philosopher->extra_fork = num_forks - num_philosophers;
    philosopher->fork_status[0] = 0;
    philosopher->fork_status[1] = 0;
    philosopher->fork_status[2] = 0;
    philosopher->num_times_think = 0;
    philosopher->num_times_eat = 0;
    philosopher->total_think_duration = 0;
    philosopher->total_eat_duration = 0;
    philosopher->min_dur = min_dur;
    philosopher->max_dur = max_dur;
    philosopher->left_fork = &forks[i];
    philosopher->right_fork = &forks[(i + 1) % num_philosophers];

    if(philosopher->extra_fork==1){
      philosopher->center_fork = &forks[num_forks-1];
    }else{
      philosopher->center_fork = NULL;
    }
    

    pthread_attr_t  attr;
    struct sched_param param;
    int policy;
    pthread_attr_init(&attr);

    if(pthread_attr_getschedpolicy(&attr, &policy) != 0)
        fprintf(stderr, "Unable to get policy.\n");

    if(pthread_attr_setschedpolicy(&attr, SCHED_RR) != 0)
      fprintf(stderr, "Unable to set policy.\n");

    pthread_attr_getschedpolicy(&attr, &policy);
    if(i==prio_phil_id){
      param.sched_priority = thread_priority_val;
      pthread_attr_setschedparam (&attr, &param);
    }else{
      param.sched_priority = 1;
      pthread_attr_setschedparam (&attr, &param);
    }

    pthread_attr_getschedparam(&attr, &param);
    printf("Thread %d Schedule Priority = %d\n", i, param.sched_priority);
    pthread_create(&threads[i], &attr, start_activity_philosopher, (void *)philosopher);
  }
}

void *start_activity_philosopher(void *arg)
{
  philosopher_t *philosopher = (philosopher_t *)arg;
  time_t start_t, end_t;
  time(&start_t);

  while(1)
  {
    think(philosopher);
    take_forks(philosopher);
    eat(philosopher);
    place_forks(philosopher);
    time(&end_t);
    
    if(difftime(end_t, start_t)*1000>END_TIME){
        sleep(1);
        print_philosopher_stats(philosopher);
        break;
    }
  }
}

void print_philosopher_stats(philosopher_t * philosopher){

  pthread_mutex_lock (&print_mutex);
  printf("--------------------------------------------\n");
  printf("Philosopher %d stats\n",philosopher->position);
  printf("Philosopher %d number of times thought: %d \n", philosopher->position, philosopher->num_times_think);
  printf("Philosopher %d number of plates eaten: %d \n", philosopher->position, philosopher->num_times_eat);
  printf("Philosopher %d total eat duration: %d ms\n", philosopher->position, philosopher->total_eat_duration);
  printf("Philosopher %d total think duration: %d ms\n", philosopher->position, philosopher->total_think_duration);
  printf("--------------------------------------------\n");
  pthread_mutex_unlock (&print_mutex);

}

void think(philosopher_t *philosopher)
{

    int think_duration; 
    think_duration = get_random_number(philosopher->min_dur, philosopher->max_dur);
    philosopher->total_think_duration += think_duration;
    //printf("Philosopher %d started thinking for %d ms\n", philosopher->position, think_duration);

    // sleep
    float microsec = get_microseconds(think_duration);
    usleep(microsec);

    philosopher->num_times_think += 1;

    //printf("Philosopher %d stopped thinking\n", philosopher->position);

}

int get_microseconds(int milliseconds){

    return milliseconds*1000;
}

void eat(philosopher_t *philosopher)
{
    int eat_time;
    eat_time = get_random_number(philosopher->min_dur, philosopher->max_dur);
    philosopher->total_eat_duration += eat_time;

    //printf("Philosopher %d started eating for %d ms\n", philosopher->position, eat_time);
    // sleep
    float microsec = get_microseconds(eat_time);
    usleep(microsec);

    philosopher->num_times_eat += 1;

    //printf("Philosopher %d stopped eating\n", philosopher->position);

}

void pick_quick_fork(philosopher_t *philosopher, sem_t *right_fork, int last){

  int right_status, center_status;

  while(1){
    
    right_status = sem_trywait(philosopher->right_fork);
    if(right_status==0){
      if(last==1)
        {philosopher->fork_status[0] = 1;}
      else
        {philosopher->fork_status[2] = 1;}
      printf("%d right status locked: %d\n", philosopher->position, right_status);
      break;
    }

    center_status = sem_trywait(philosopher->center_fork);
    if(center_status==0){
      philosopher->fork_status[1] = 1;
      printf("%d center status locked: %d\n", philosopher->position, center_status);
      break;
    }
  }
}

void take_forks(philosopher_t *philosopher)
{ 
  if (is_last_philosopher(philosopher))
  {
      sem_wait(philosopher->left_fork);
      philosopher->fork_status[0] = 1;
      if (philosopher->center_fork == NULL){
        sem_wait(philosopher->right_fork);
        philosopher->fork_status[2] = 1;
      }else{
        pick_quick_fork(philosopher, philosopher->right_fork, 0);
      }
  }
  else
  {
      sem_wait(philosopher->right_fork);
      philosopher->fork_status[2] = 1;
      if (philosopher->center_fork == NULL){
        sem_wait(philosopher->left_fork);
        philosopher->fork_status[0] = 1;
      }else{
        pick_quick_fork(philosopher, philosopher->left_fork, 1);
      }
  }
}

void place_forks(philosopher_t *philosopher)
{   
    if(philosopher->fork_status[2]==1){
      sem_post(philosopher->right_fork);
      philosopher->fork_status[2]==0;
      printf("right fork status\n");
    }
    
    if(philosopher->fork_status[0]==1){
      sem_post(philosopher->left_fork);
      philosopher->fork_status[0]==0;
      printf("left fork status\n");
    }

    if(philosopher->fork_status[1]==1){
      sem_post(philosopher->center_fork);
      philosopher->fork_status[1]==0;
      printf("center fork status\n");
    }

    printf("%d left fork status: %d\n", philosopher->position, philosopher->fork_status[0]);
    printf("%d center fork status: %d\n", philosopher->position, philosopher->fork_status[1]);
    printf("%d right fork status: %d\n", philosopher->position, philosopher->fork_status[2]);

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
