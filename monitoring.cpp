#include <string>
#include <fstream>
#include <sys/time.h>
#include <cmath>

#include "global_phs.h"
#include "monitoring.h"
#include "file_input.h"

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

void compare_phs_gpu_vs_ref (FILE *fp, int n_events, int n_channels, int *channels,
                             phs_val_t *pval,
                             double *pgen, double *factors, double *volumes) {
   int n_events_failed = 0;
   double epsilon = input_control.compare_tolerance;
   double max_abs_deviation[4] = {0, 0, 0, 0};
   double sum_abs_deviation[4] = {0, 0, 0, 0};
   for (int i = 0; i < n_events; i++) {
      int c = channels[i];
      if (contains_friends[c] == 0) {
         for (int n = 0; n < N_EXT_OUT; n++) {
            double *pv = pval[i].prt[N_EXT_IN+n].p;
            double *pg = &pgen[4*N_EXT_OUT*i + 4*n];
            if (fabs(pv[0] - pg[0]) > epsilon || fabs(pv[1] - pg[1]) > epsilon 
             || fabs(pv[2] - pg[2]) > epsilon || fabs(pv[3] - pg[3]) > epsilon) {
               fprintf (fp, "Error in p%d: (event: %d, channel: %d):\n", N_EXT_IN + n + 1, i, c);
               fprintf (fp, "Validation: %lf %lf %lf %lf\n", pv[0], pv[1], pv[2], pv[3]);
               fprintf (fp, "Generated:  %lf %lf %lf %lf\n", pg[0], pg[1], pg[2], pg[3]);
               for (int j = 0; j < 4; j++) {
                  double d = fabs(pv[j] - pg[j]); 
                  if (d > max_abs_deviation[j]) max_abs_deviation[j] = d;
                  sum_abs_deviation[j] += d;
               }
               n_events_failed++;
            }
         }
  
         if (input_control.do_inverse_mapping) {
            for (int oc = 0; oc < n_channels; oc++) {
               double gpu_factor = factors[n_channels * i + oc];
               double ref_factor = pval[i].factors[oc];
               if (fabs (gpu_factor - ref_factor) > epsilon && !contains_friends[oc]) {
                  fprintf (fp, "Error in factor (%d,%d): Validation: %lf, Generated: %lf\n", i, oc, ref_factor, gpu_factor);
                  n_events_failed++;
               }
            }
         }

         if (fabs (pval[i].v - volumes[i]) > epsilon) {
            fprintf (fp, "Error in volume (%d): Validation: %lf, Generated: %lf\n", i, pval[i].v, volumes[i]);
            n_events_failed++;
         }
      } else if (contains_friends[c] == 1) {
          fprintf (fp, "Channel %d contains friends. Skip\n", c);
          contains_friends[c] = -1;
      }
   }
   fprintf (fp, "Failed events with EPSILON = %lf: %d / %d (%.2f%%)\n", epsilon, n_events_failed, n_events, (double)n_events_failed / n_events * 100);
   if (n_events_failed > 0) {
      fprintf (fp, "Max. deviations: %lf %lf %lf %lf\n",
               max_abs_deviation[0], max_abs_deviation[1], max_abs_deviation[2], max_abs_deviation[3]);
      fprintf (fp, "Avg. deviations: %lf %lf %lf %lf\n",
               sum_abs_deviation[0] / n_events_failed, sum_abs_deviation[1] / n_events_failed,
               sum_abs_deviation[2] / n_events_failed, sum_abs_deviation[3] / n_events_failed);
   }
}

void compare_phs_cpu_vs_ref (FILE *fp, int n_events_val, int n_events_gen,
                             int n_channels, int *channels, phs_val_t *pval,
                             phs_prt_t *prt, double *factors, double *volumes) {
   int n_events_failed = 0;
   double epsilon = input_control.compare_tolerance;
   double max_abs_deviation[4] = {0, 0, 0, 0};
   double sum_abs_deviation[4] = {0, 0, 0, 0};
   for (int i = 0; i < n_events_val; i++) {
      int c = channels[i];
      for (int n = 0; n < N_EXT_OUT; n++) {
         double *p = pval[i].prt[N_EXT_IN+n].p;
         int nn = pow(2,n) - 1;
         if (fabs (p[0] - prt[N_PRT*i + nn].p[0]) > epsilon
          || fabs (p[1] - prt[N_PRT*i + nn].p[1]) > epsilon
          || fabs (p[2] - prt[N_PRT*i + nn].p[2]) > epsilon
          || fabs (p[3] - prt[N_PRT*i + nn].p[3]) > epsilon) {
               fprintf (fp, "Error in p%d (event: %d, channel: %d):\n", n, i, c);
               fprintf (fp, "Validation: %lf %lf %lf %lf\n", p[0], p[1], p[2], p[3]);
               fprintf (fp, "Generated:  %lf %lf %lf %lf\n", prt[N_PRT*i + nn].p[0], prt[N_PRT*i + nn].p[1],
                                                        prt[N_PRT*i + nn].p[2], prt[N_PRT*i + nn].p[3]);
               for (int j = 0; j < 4; j++) {
                  double d = fabs(p[j] - prt[N_PRT*i + nn].p[j]); 
                  if (d > max_abs_deviation[j]) max_abs_deviation[j] = d;
                  sum_abs_deviation[j] += d;
               }
               n_events_failed++;
         }

         if (input_control.do_inverse_mapping) {
            for (int oc = 0; oc < n_channels; oc++) {
               double cpu_factor = factors[n_channels * i + oc];
               double ref_factor = pval[i].factors[oc];
               if (fabs (ref_factor - cpu_factor) > epsilon) {
                  fprintf (fp, "Error in factor (%d,%d): Validation: %lf, Generated: %lf\n", i, oc, ref_factor, cpu_factor);
                  n_events_failed++;
               }
            }
         }

         if (fabs (pval[i].v - volumes[i]) > epsilon) {
            fprintf (fp, "Error in volume (%d): Validation: %lf, Generated: %lf\n", i, pval[i].v, volumes[i]);
            n_events_failed++;
         }
     }
   }
   fprintf (fp, "Failed events with EPSILON = %lf: %d / %d\n", epsilon, n_events_failed, n_events_gen);
}

size_t required_gpu_mem (size_t n_events, int n_x, int n_channels) {
   size_t mem = 0;
   // Random numbers and counter indices
   mem += n_x * n_events * sizeof(double);
   mem += n_events * sizeof(int);
   // Commands are negligible
   // Momenta
   mem += N_PRT * n_events * 4 * sizeof(double);
   // Kinematic scratchpads (msq, p_decay, boosts);
   mem += 2 * N_BRANCHES * n_events * sizeof(double);
   mem += 16 * N_BOOSTS * n_events * sizeof(double);
   // Channel ids
   mem += n_events * sizeof(int);
   // local factors & volumes
   mem += 2 * N_BRANCHES * n_events * sizeof(double);
   // global factors
   if (input_control.do_inverse_mapping) mem += n_channels * n_events * sizeof(double);
   // oks
   mem += n_events * sizeof(bool);
   return mem;
}

// In short:  mem = (n_x * sizeof(double) + 2 * sizeof(int) 
//                +  4 * N_BRANCHES * sizeof(double)
//                +  16 * N_BOOSTS * sizeof(double)
//                +  4 * N_PRT * sizeof(double)
//                +  sizeof(bool)) * n_events 
//
// With inverse mappings: += n_channels * n_events * sizeof(double)
size_t nevents_that_fit_into_gpu_mem (size_t mem, int n_x, int n_channels) {
   size_t mem_per_element = n_x * sizeof(double) + 2 * sizeof(int)
                             + 4 * N_BRANCHES * sizeof(double)
                             + 16 * N_BOOSTS * sizeof(double)
                             + 4 * N_PRT * sizeof(double)
                             + sizeof(bool);
   if (input_control.do_inverse_mapping) mem_per_element += n_channels * sizeof(double);
   return mem / mem_per_element / n_channels;
}

size_t required_cpu_mem (size_t n_events, int n_x) {
   size_t mem = 0;
   // Random numbers
   mem += n_x * n_events * sizeof(double);
   // prt
   mem += 4 * N_PRT * sizeof(double);
   return mem;
}
