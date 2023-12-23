#ifndef MONITORING_H
#define MONITORING_H

#include <stdio.h>

#include "phs.h"


enum {TIME_KERNEL_MSQ = 0,
      TIME_KERNEL_CB = 1,
      TIME_KERNEL_AB = 2,
      TIME_KERNEL_INIT = 3,
      TIME_MEMCPY_IN = 4,
      TIME_MEMCPY_OUT = 5};

enum {SAFE_CUDA_INIT = 0,
      SAFE_CUDA_MSQ = 1,
      SAFE_CUDA_CB = 2,
      SAFE_CUDA_AB = 3};
extern const char *cuda_checks[4];

enum {LOG_INPUT = 0,
      LOG_CUDA = 1};
      
extern double gpu_timers[6];
extern FILE *logfl[2];

double mysecond ();
void init_monitoring (char *filename_input, char *filename_cuda);
void final_monitoring ();

#define START_TIMER(id) \
do { \
   gpu_timers[id] -= mysecond(); \
} while (0)

#define STOP_TIMER(id) \
do { \
   gpu_timers[id] += mysecond(); \
} while (0)

#define CHECK_CUDA_STATE(id) \
do { \
   fprintf (logfl[LOG_CUDA], "%s: %s\n", cuda_checks[id], cudaGetErrorString(cudaGetLastError())); \
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
