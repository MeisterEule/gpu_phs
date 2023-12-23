#include <string>
#include <fstream>
#include <sys/time.h>
#include <cmath>

#include "monitoring.h"

double gpu_timers[6] = {0, 0, 0, 0, 0, 0};
const char *cuda_checks[4] = {"Init", "MSQ", "Create Boosts", "Apply Boosts"};

FILE *logfl[2] = {NULL, NULL};

double mysecond () {
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp,&tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void init_monitoring (char *filename_input, char *filename_cuda) {
   logfl[LOG_INPUT] = fopen (filename_input, "w+");
   logfl[LOG_CUDA] = fopen (filename_cuda, "w+");
}

void final_monitoring () {
   for (int i = 0; i < 2; i++) {
      fclose(logfl[i]);
      logfl[i] = NULL;
   }
}

#define EPSILON 0.0001

void compare_phs_gpu_vs_ref (FILE *fp, int n_events, int *channels,
                             int n_in, int n_out, phs_val_t *pval,
                             double *pgen, double *factors, double *volumes) {
   int n_events_failed = 0;
   for (int i = 0; i < n_events; i++) {
      for (int n = 0; n < n_out; n++) {
         double *pv = pval[i].prt[n_in+n].p;
         double *pg = &pgen[4*n_out*i + 4*n];
         if (fabs(pv[0] - pg[0]) > EPSILON || fabs(pv[1] - pg[1]) > EPSILON 
          || fabs(pv[2] - pg[2]) > EPSILON || fabs(pv[3] - pg[3]) > EPSILON) {
            fprintf (fp, "Error in p%d: (event: %d, channel: %d):\n", n_in + n + 1, i, channels[i]);
            fprintf (fp, "Validation: %lf %lf %lf %lf\n", pv[0], pv[1], pv[2], pv[3]);
            fprintf (fp, "Generated:  %lf %lf %lf %lf\n", pg[0], pg[1], pg[2], pg[3]);
            n_events_failed++;
         }
      }
  
      if (fabs (pval[i].f - factors[i]) > EPSILON) {
         fprintf (fp, "Error in factor (%d): Validation: %lf, Generated: %lf\n", i, pval[i].f, factors[i]);
         n_events_failed++;
      }

      if (fabs (pval[i].v - volumes[i]) > EPSILON) {
         fprintf (fp, "Error in factor (%d): Validation: %lf, Generated: %lf\n", i, pval[i].v, volumes[i]);
         n_events_failed++;
      }
   }
   fprintf (fp, "Failed events with EPSILON = %lf: %d / %d\n", EPSILON, n_events_failed, n_events);
}

void compare_phs_cpu_vs_ref (FILE *fp, int n_events_val, int n_events_gen,
                             int *channels, int n_in, int n_out, phs_val_t *pval,
                             phs_prt_t *prt, double *factors, double *volumes) {
   int n_events_failed = 0;
   for (int i = 0; i < n_events_val; i++) {
      for (int n = 0; n < n_out; n++) {
         double *p = pval[i].prt[n_in+n].p;
         int nn = pow(2,n) - 1;
         if (fabs (p[0] - prt[N_PRT*i + nn].p[0]) > EPSILON
          || fabs (p[1] - prt[N_PRT*i + nn].p[1]) > EPSILON
          || fabs (p[2] - prt[N_PRT*i + nn].p[2]) > EPSILON
          || fabs (p[3] - prt[N_PRT*i + nn].p[3]) > EPSILON) {
               fprintf (fp, "Error in p%d (event: %d, channel: %d):\n", n, i, channels[i]);
               fprintf (fp, "Validation: %lf %lf %lf %lf\n", p[0], p[1], p[2], p[3]);
               fprintf (fp, "Generated:  %lf %lf %lf %lf\n", prt[N_PRT*i + nn].p[0], prt[N_PRT*i + nn].p[1],
                                                        prt[N_PRT*i + nn].p[2], prt[N_PRT*i + nn].p[3]);
               n_events_failed++;
         }

         if (fabs (pval[i].f - factors[i]) > EPSILON) {
            fprintf (fp, "Error in factor (%d): Validation: %lf, Generated: %lf\n", i, pval[i].f, factors[i]);
            n_events_failed++;
         }

         if (fabs (pval[i].v - volumes[i]) > EPSILON) {
            fprintf (fp, "Error in volume (%d): Validation: %lf, Generated: %lf\n", i, pval[i].v, volumes[i]);
            n_events_failed++;
         }
     }
   }
   fprintf (fp, "Failed events with EPSILON = %lf: %d / %d\n", EPSILON, n_events_failed, n_events_gen);
}

long long required_gpu_mem (long long n_events, int n_x) {
   long long mem = 0;
   // Random numbers and counter indices
   mem += n_x * n_events * sizeof(double);
   mem += n_events * sizeof(int);
   // Commands are negligible
   // Momenta
   mem += N_BRANCHES * n_events * 4 * sizeof(double);
   // Kinematic scratchpads (msq, p_decay, boosts);
   mem += 2 * N_BRANCHES * n_events * sizeof(double);
   mem += 16 * N_BOOSTS * n_events * sizeof(double);
   // Channel ids
   mem += n_events * sizeof(int);
   // factors & volumes
   mem += 2 * N_BRANCHES * n_events * sizeof(double);
   // oks
   mem += n_events * sizeof(bool);
   return mem;
}

// In short:  mem = (n_x * sizeof(double) + 2 * sizeof(int) 
//                +  10 * N_BRANCHES * sizeof(double)
//                +  16 * N_BOOSTS * sizeof(double)
//                +  sizeof(bool)) * n_events 
//
long long nevents_that_fit_into_gpu_mem (long long mem, int n_x, int n_channels) {
   long long mem_per_element = n_x * sizeof(double) + 2 * sizeof(int)
                             + 8 * N_BRANCHES * sizeof(double)
                             + 16 * N_BOOSTS * sizeof(double)
                             + sizeof(bool);
   return mem / mem_per_element / n_channels;
}

long long required_cpu_mem (long long n_events, int n_x) {
   long long mem = 0;
   // Random numbers
   mem += n_x * n_events * sizeof(double);
   // prt
   mem += 4 * N_PRT * sizeof(double);
   return mem;
}
