#ifndef MONITORING_H
#define MONITORING_H

#include <stdio.h>

enum {TIME_KERNEL_MSQ = 0,
      TIME_KERNEL_ANG = 1,
      TIME_MEMCPY_IN = 2,
      TIME_MEMCPY_OUT = 3,
      TIME_MEMCPY_BOOST = 4}; 

enum {LOG_INPUT = 0,
      LOG_CUDA = 1};
      
extern double gpu_timers[5];
extern FILE *logfl[2];

double mysecond ();
void init_logfiles (char *filename_input, char *filename_cuda);
void final_logfiles ();

#define START_TIMER(id) \
do { \
   gpu_timers[id] -= mysecond(); \
} while (0)

#define STOP_TIMER(id) \
do { \
   gpu_timers[id] += mysecond(); \
} while (0)

#endif
