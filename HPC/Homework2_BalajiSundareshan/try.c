#define _MULTI_THREADED
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
//#include "check.h"

#define  BUMP_PRIO     1
int      thePriority = 0;

int showSchedParam(pthread_t thread)
{
  struct sched_param   param;
  int                  policy;
  int                  rc;

  printf("Get scheduling parameters\n");
  rc = pthread_getschedparam(thread, &policy, &param);
  checkResults("pthread_getschedparam()\n", rc);

  printf("The thread scheduling parameters indicate:\n"
         "priority = %d\n", param.sched_priority);
  return param.sched_priority;
}

void *threadfunc(void *parm)
{
  int           rc;

  printf("Inside secondary thread\n");
  thePriority = showSchedParam(pthread_self());
  sleep(5);  /* Sleep is not a very robust way to serialize threads */
  return NULL;
}

int main(int argc, char **argv)
{
  pthread_t             thread;
  int                   rc=0;
  struct sched_param    param;
  int                   policy = SCHED_OTHER;
  int                   theChangedPriority=0;

  printf("Enter Testcase - %s\n", argv[0]);

  printf("Create thread using default attributes\n");
  rc = pthread_create(&thread, NULL, threadfunc, NULL);
  checkResults("pthread_create()\n", rc);

  sleep(2);  /* Sleep is not a very robust way to serialize threads */

  memset(&param, 0, sizeof(param));
  /* Bump the priority of the thread a small amount */
  if (thePriority - BUMP_PRIO >= PRIORITY_MIN_NP) {
      param.sched_priority = thePriority - BUMP_PRIO;
  }

  printf("Set scheduling parameters, prio=%d\n",
         param.sched_priority);
  rc = pthread_setschedparam(thread, policy, &param);
  checkResults("pthread_setschedparam()\n", rc);

  /* Let the thread fill in its own last priority */
  theChangedPriority = showSchedParam(thread);
 
  if (thePriority == theChangedPriority ||
      param.sched_priority != theChangedPriority) {
      printf("The thread did not get priority set correctly, "
             "first=%d last=%d expected=%d\n",
             thePriority, theChangedPriority, param.sched_priority);
      exit(1);
  }
  
  sleep(5);  /* Sleep is not a very robust way to serialize threads */
  printf("Main completed\n");
  return 0;
}
