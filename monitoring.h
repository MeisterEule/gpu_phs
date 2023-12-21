#ifndef MONITORING_H
#define MONITORING_H

#include <stdio.h>

#include "phs.h"


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

void compare_phs_gpu_vs_ref (FILE *fp, int n_events, int *channels,
                             int n_in, int n_out, phs_val_t *pval,
                             double *pgen, double *factors, double *volumes);

void compare_phs_cpu_vs_ref (FILE *fp, int n_events_val, int n_events_gen,
                             int *channels, int n_in, int n_out, phs_val_t *pval,
                             phs_prt_t *prt, double *factors, double *volumes);

long long required_gpu_mem (long long n_events, int n_x);
long long required_cpu_mem (long long n_events, int n_x);

long long nevents_that_fit_into_gpu_mem (long long mem, int n_x, int n_channels);

#endif
