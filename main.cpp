#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <math.h>
#include <vector>
#include <algorithm>
#include <cstring>

#include "rng.h"
#include "monitoring.h"
#include "phs.h"
#include "mom_generator.h"

bool verify_against_whizard = false;

void do_verify_against_whizard (char *ref_file, int n_x, int n_channels, 
                                int n_in, int n_out, int filepos_start_mom) {
   int n_events = count_nevents_in_reference_file (ref_file, n_in + n_out, filepos_start_mom);

   fprintf (logfl[LOG_INPUT], "n_events in reference file: %lld\n", n_events);

   double sqrts = 1000;
   double *x = (double*)malloc(n_x * n_events * sizeof(double));

   phs_val_t *pval = (phs_val_t*)malloc(n_events * sizeof (phs_val_t));
   for (int i = 0; i < n_events; i++) {
      pval[i].prt = (phs_prt_t*)malloc((n_in + n_out) * sizeof(phs_prt_t));
   }

   int *channel_lims = (int*)malloc((n_channels + 1) * sizeof(int));
   channel_lims[0] = 0;
   channel_lims[n_channels] = n_events;
   read_reference_momenta (ref_file, filepos_start_mom, n_in + n_out, n_x, x, channel_lims, pval);

   int *channels = (int*)malloc(n_events * sizeof(int));
   int c = 0;
   for (int i = 0; i < n_events; i++) {
      if (i == channel_lims[c+1]) c++;
      channels[i] = c;
   }

   long long mem_gpu = required_gpu_mem (n_events, n_x);
   long long mem_cpu = required_cpu_mem (n_events, n_x);
   printf ("Required GPU memory: %lf GiB\n", (double)mem_gpu / BYTES_PER_GB);
   printf ("Required CPU memory: %lf GiB\n", (double)mem_cpu / BYTES_PER_GB);

   fprintf (logfl[LOG_INPUT], "channel_limits: ");
   for (int i = 0; i < n_channels + 1; i++) {
      fprintf (logfl[LOG_INPUT], "%d ", channel_lims[i]);
   }
   fprintf (logfl[LOG_INPUT], "\n");

   double *p = (double*)malloc(4 * n_out * n_events * sizeof(double));
   double *factors = (double*)malloc(n_events * sizeof(double)); 
   double *volumes = (double*)malloc(n_events * sizeof(double)); 
   bool *oks = (bool*)malloc(N_PRT * n_events * sizeof(bool));

   init_mapping_constants_cpu (n_channels, sqrts * sqrts, 0, sqrts * sqrts);
   init_phs_gpu(n_channels, mappings_host, sqrts * sqrts);
   double t1 = mysecond();
   gen_phs_from_x_gpu (n_events, n_channels, channels, n_x, x, factors, volumes, oks, p);
   double t2 = mysecond();

   printf ("GPU: %lf sec\n", t2 - t1);
   FILE *fp = fopen ("compare.gpu", "w+");
   compare_phs_gpu_vs_ref (fp, n_events, channels, n_in, n_out, pval, p, factors, volumes);
   fclose(fp);
   fp = NULL;


   free (p);
   free (factors);
   free (volumes);
  
   phs_prt_t *prt = (phs_prt_t*)malloc(N_PRT * n_events * sizeof(phs_prt_t));
   factors = (double*)malloc(n_events * sizeof(double));
   volumes = (double*)malloc(n_events * sizeof(double));
   oks = (bool*)malloc(n_events * sizeof(bool));

   t1 = mysecond();
   gen_phs_from_x_cpu (sqrts, n_events, n_x, x, channels, factors, volumes, oks, prt);
   t2 = mysecond();

   fp = fopen ("compare.cpu", "w+");
   compare_phs_cpu_vs_ref (fp, n_events, n_events,
                           channels, n_in, n_out, pval, prt, factors, volumes);
   fclose(fp);
   fp = NULL;

   printf ("CPU: %lf sec\n", t2 - t1);


   free (factors);
   free (volumes);
   free (oks);
   free (prt);
   free (x);
   free (pval);
}

void do_verify_internal (int n_events_per_channel, int n_trials, int n_trial_events,
                         int n_x, int n_channels, int n_in, int n_out) {
   double t1, t2;
   int n_events  = n_events_per_channel * n_channels;
   int n_trial_events_tot = n_trial_events * n_channels;

   double sqrts = 1000;
   init_mapping_constants_cpu (n_channels, sqrts * sqrts, 0, sqrts * sqrts);

   double *x = (double*)malloc(n_x * n_events * sizeof(double));

   init_rng (n_channels, n_x);

   int *channels = (int*)malloc(n_trial_events_tot * sizeof(int));
   for (int i = 0; i < n_trial_events_tot; i++) {
      channels[i] = i / n_trial_events;
   }

   double *p = (double*)malloc(4 * n_out * n_events * sizeof(double));
   double *factors = (double*)malloc(n_events * sizeof(double)); 
   double *volumes = (double*)malloc(n_events * sizeof(double)); 
   bool *oks_gpu = (bool*)malloc(n_events * sizeof(bool));
   bool *oks_cpu = (bool*)malloc(n_events * sizeof(bool));

   init_mapping_constants_cpu (n_channels, sqrts * sqrts, 0, sqrts * sqrts);
   init_phs_gpu(n_channels, mappings_host, sqrts * sqrts);

   int n_ok;
   printf ("Precondition grid with %d trials and %d events / trial.\n", n_trials, n_trial_events);
   // Assert n_trial_events <= n_events
   for (int i = 0; i < n_trials; i++) {
      rng_generate (n_channels, n_x, n_trial_events, x);
      gen_phs_from_x_gpu (n_trial_events_tot, n_channels, channels,
                          n_x, x, factors, volumes, oks_gpu, p);
      n_ok = 0;
      for (int i = 0; i < n_trial_events_tot; i++) {
        if (oks_gpu[i]) n_ok++;
      }
      printf ("Trial %d: %d / %d\n", i, n_ok, n_trial_events_tot);

      update_weights (n_x, n_channels, n_trial_events_tot, channels, x, oks_gpu);
   }

   free(channels);
   channels = (int*)malloc(n_events * sizeof(int));
   for (int i = 0; i < n_events; i++) {
     channels[i] = i / n_events_per_channel;
   }


   // Now do the real time measurement with the adapted grids
   printf ("Perform optimized GPU run with %d events:\n", n_events);
   printf ("Required GPU memory: %lf GiB\n", (double)required_gpu_mem (n_events, n_x) / BYTES_PER_GB);
   printf ("Required CPU memory: %lf GiB\n", (double)required_cpu_mem (n_events, n_x) / BYTES_PER_GB);

   rng_generate (n_channels, n_x, n_events_per_channel, x);
   t1 = mysecond();
   gen_phs_from_x_gpu (n_events, n_channels, channels, n_x, x, factors, volumes, oks_gpu, p);
   t2 = mysecond();

   printf ("GPU: %lf sec\n", t2 - t1);
   n_ok = 0;
   for (int i = 0; i < n_events; i++) {
     if (oks_gpu[i]) n_ok++;
   }
   printf ("Valid events: %d / %d\n", n_ok, n_events);

   free(p);
   phs_prt_t *prt = (phs_prt_t*)malloc(N_PRT * n_events * sizeof(phs_prt_t));
   memset (prt, 0, N_PRT * 4 * n_events * sizeof(double));
   ///double *factors = (double*)malloc(d.n_events_gen * sizeof(double));
   ///double *volumes = (double*)malloc(d.n_events_gen * sizeof(double));
   //
   //for (int i = 0; i < d.n_events_gen; i++) {
   //   oks[i] = true;
   //   factors[i] = 1;
   //   volumes[i] = 1;
   //}

   t1 = mysecond();
   gen_phs_from_x_cpu (sqrts, n_events, n_x, x, channels, factors, volumes, oks_cpu, prt);
   t2 = mysecond();

   printf ("CPU: %lf sec\n", t2 - t1);

   n_ok = 0;
   for (int i = 0; i < n_events; i++) {
     if (oks_cpu[i]) n_ok++;
   }
   printf ("Valid events: %d / %d\n", n_ok, n_events);


   for (int i = 0; i < n_events; i++) {
      if (oks_cpu[i] != oks_gpu[i]) {
         if (oks_cpu[i]) {
           printf ("CPU OK, GPU FAIL: %d\n", i);
         } else {
           printf ("GPU OK, CPU FAIL: %d\n", i);
         }
      }
   }
}

int main (int argc, char *argv[]) {
   if (argc < 2) {
      printf ("No reference file given!\n");
      return -1;
   }
   char *ref_file = argv[1];
   // Check that the ref file exists
   //
   init_logfiles ("input.log", "cuda.log");

   int n_prt_tot, n_prt_out;
   int filepos = 0;
   int *header_data = (int*)malloc (NHEADER * sizeof(int));
   read_reference_header (ref_file, header_data, &filepos);
   int n_channels = header_data[H_NCHANNELS];
   int n_in = header_data[H_NIN];
   int n_out = header_data[H_NOUT];
   int n_trees = header_data[H_NTREES];
   int n_forests = header_data[H_NGROVES];
   int n_x = header_data[H_NX];

   N_PRT = pow(2,n_in + n_out) - 1;
   N_PRT_OUT = pow(2, n_out) - 1;
   PRT_STRIDE = 4 * N_PRT;
   ROOT_BRANCH = N_PRT_OUT - 1;

   N_BRANCHES = 2*n_out - 1;
   N_BRANCHES_INTERNAL = N_BRANCHES - n_out;
   N_MSQ = n_out - 2;
   N_LAMBDA_IN = n_out - 1;
   N_BOOSTS = N_LAMBDA_IN + 1;
   N_LAMBDA_OUT = N_BRANCHES;

   fprintf (logfl[LOG_INPUT], "n_channels: %d\n", n_channels);
   fprintf (logfl[LOG_INPUT], "n_in: %d\n", n_in);
   fprintf (logfl[LOG_INPUT], "n_out: %d\n", n_out);
   fprintf (logfl[LOG_INPUT], "n_channels: %d\n", n_channels);
   fprintf (logfl[LOG_INPUT], "nx: %d\n", n_x);
   fprintf (logfl[LOG_INPUT], "NPRT: %d\n", N_PRT);
   fprintf (logfl[LOG_INPUT], "NPRT_OUT: %d\n", N_PRT_OUT);

   daughters1 = (int**)malloc(n_channels * sizeof(int*));
   daughters2 = (int**)malloc(n_channels * sizeof(int*));
   has_children = (int**)malloc(n_channels * sizeof(int*));
   mappings_host = (mapping_t*)malloc(n_channels * sizeof(mapping_t));
   for (int i = 0; i < n_channels; i++) {
      daughters1[i] = (int*)malloc(N_PRT * sizeof(int));
      daughters2[i] = (int*)malloc(N_PRT * sizeof(int));
      has_children[i] = (int*)malloc(N_PRT * sizeof(int));
      mappings_host[i].map_id = (int*)malloc(N_PRT_OUT * sizeof(int));
      mappings_host[i].comp_msq = NULL;
      mappings_host[i].comp_ct = NULL;
      mappings_host[i].a = (map_constant_t*)malloc(N_PRT_OUT * sizeof(map_constant_t));
      mappings_host[i].b = (map_constant_t*)malloc(N_PRT_OUT * sizeof(map_constant_t));
      mappings_host[i].masses = (double*)malloc(N_PRT_OUT * sizeof(double));
      mappings_host[i].widths = (double*)malloc(N_PRT_OUT * sizeof(double));
   }
   read_tree_structures (ref_file, n_channels, N_PRT, N_PRT_OUT, &filepos);

   for (int c = 0; c < n_channels; c++) {
      fprintf (logfl[LOG_INPUT], "Channel %d: \n", c);
      fprintf (logfl[LOG_INPUT], "daughters1: ");
      for (int i = 0; i < N_PRT; i++) {
         daughters1[c][i]++;
         fprintf (logfl[LOG_INPUT], "%d ", daughters1[c][i]);
      }
      fprintf (logfl[LOG_INPUT], "\ndaughters2: ");
      for (int i = 0; i < N_PRT; i++) {
         daughters2[c][i]++;
         fprintf (logfl[LOG_INPUT], "%d ", daughters2[c][i]);
      }
      fprintf (logfl[LOG_INPUT], "\nhas_children: ");
      for (int i = 0; i < N_PRT; i++) {
         fprintf (logfl[LOG_INPUT], "%d ", has_children[c][i]);
      }
      fprintf (logfl[LOG_INPUT], "\nmappings: ");
      for (int i = 0; i < N_PRT_OUT; i++) {
         fprintf (logfl[LOG_INPUT], "%d ", mappings_host[c].map_id[i]);
      }
      fprintf (logfl[LOG_INPUT], "\nmasses: ");
      for (int i = 0; i < N_PRT_OUT; i++) {
         fprintf (logfl[LOG_INPUT], "%lf ", mappings_host[c].masses[i]);
      }
      fprintf (logfl[LOG_INPUT], "\nwidths: ");
      for (int i = 0; i < N_PRT_OUT; i++) {
         fprintf (logfl[LOG_INPUT], "%lf ", mappings_host[c].widths[i]);
      }
      fprintf (logfl[LOG_INPUT], "\n");
   }


    
   if (verify_against_whizard) {
      do_verify_against_whizard (ref_file, n_x, n_channels, n_in, n_out, filepos);
   } else {
      do_verify_internal (1000, 10, 1000, n_x, n_channels, n_in, n_out);
   }

   final_logfiles();
   return 0;
}
