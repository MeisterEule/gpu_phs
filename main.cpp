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

void do_verify_against_whizard (char *ref_file, int n_x, int n_trees, 
                                int n_in, int n_out, int filepos_start_mom) {
   phs_dim_t d;
   d.n_events_val = count_nevents_in_reference_file (ref_file, n_in + n_out, filepos_start_mom);

   fprintf (logfl[LOG_INPUT], "n_events in reference file: %lld\n", d.n_events_val);

   double sqrts = 1000;
   double *x = (double*)malloc(n_x * d.n_events_val * sizeof(double));

   phs_val_t *pval = (phs_val_t*)malloc(d.n_events_val * sizeof (phs_val_t));
   for (int i = 0; i < d.n_events_val; i++) {
      pval[i].prt = (phs_prt_t*)malloc((n_in + n_out) * sizeof(phs_prt_t));
   }

   int *channel_lims = (int*)malloc((n_trees + 1) * sizeof(int));
   channel_lims[0] = 0;
   channel_lims[n_trees] = d.n_events_val;
   read_reference_momenta (ref_file, filepos_start_mom, n_in + n_out, n_x, x, channel_lims, pval);
   d.n_events_gen = d.n_events_val;

   int *channels = (int*)malloc(d.n_events_gen * sizeof(int));
   int c = 0;
   for (int i = 0; i < d.n_events_gen; i++) {
      if (i == channel_lims[c+1]) c++;
      channels[i] = c;
   }

   long long mem_gpu = required_gpu_mem (d, n_x);
   long long mem_cpu = required_cpu_mem (d, n_x);
   printf ("Required GPU memory: %lf GiB\n", (double)mem_gpu / BYTES_PER_GB);
   printf ("Required CPU memory: %lf GiB\n", (double)mem_cpu / BYTES_PER_GB);

   fprintf (logfl[LOG_INPUT], "channel_limits: ");
   for (int i = 0; i < n_trees + 1; i++) {
      fprintf (logfl[LOG_INPUT], "%d ", channel_lims[i]);
   }
   fprintf (logfl[LOG_INPUT], "\n");
   fprintf (logfl[LOG_INPUT], "n_events to generate: %lld\n", d.n_events_gen);
   fprintf (stdout, "n_events to generate: %lld\n", d.n_events_gen);

   //long long mem_gpu = count_gpu_memory_requirements (d, n_x);

   double *p = (double*)malloc(4 * n_out * d.n_events_gen * sizeof(double));
   double *factors = (double*)malloc(d.n_events_gen * sizeof(double)); 
   double *volumes = (double*)malloc(d.n_events_gen * sizeof(double)); 
   bool *oks = (bool*)malloc(N_PRT * d.n_events_gen * sizeof(bool));

   init_mapping_constants_cpu (n_trees, sqrts * sqrts, 0, sqrts * sqrts);
   init_phs_gpu(n_trees, mappings_host, sqrts * sqrts);
   double t1 = mysecond();
   gen_phs_from_x_gpu (d.n_events_gen, n_trees, channels, n_x, x, factors, volumes, oks, p);
   double t2 = mysecond();

   printf ("GPU: %lf sec\n", t2 - t1);
   FILE *fp = fopen ("compare.gpu", "w+");
   compare_phs_gpu_vs_ref (fp, d.n_events_gen, channels, n_in, n_out, pval, p, factors, volumes);
   fclose(fp);
   fp = NULL;


   free (p);
   free (factors);
   free (volumes);
  
   phs_prt_t *prt = (phs_prt_t*)malloc(N_PRT * d.n_events_gen * sizeof(phs_prt_t));
   factors = (double*)malloc(d.n_events_gen * sizeof(double));
   volumes = (double*)malloc(d.n_events_gen * sizeof(double));
   oks = (bool*)malloc(d.n_events_gen * sizeof(bool));

   t1 = mysecond();
   gen_phs_from_x_cpu (sqrts, d, n_x, x, channels, factors, volumes, oks, prt);
   t2 = mysecond();

   fp = fopen ("compare.cpu", "w+");
   compare_phs_cpu_vs_ref (fp, d.n_events_val, d.n_events_gen,
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
   phs_dim_t d;
   d.n_events_gen = n_events_per_channel * n_channels;
   d.n_events_val = n_events_per_channel * n_channels;

   long long mem_gpu = required_gpu_mem (d, n_x);
   long long mem_cpu = required_cpu_mem (d, n_x);
   printf ("Required GPU memory: %lf GiB\n", (double)mem_gpu / BYTES_PER_GB);
   printf ("Required CPU memory: %lf GiB\n", (double)mem_cpu / BYTES_PER_GB);

   double sqrts = 1000;
   init_mapping_constants_cpu (n_channels, sqrts * sqrts, 0, sqrts * sqrts);

   printf ("Allocate x: %d * %d = %d\n", n_x, d.n_events_gen, n_x * d.n_events_gen);
   double *x = (double*)malloc(n_x * d.n_events_gen * sizeof(double));

   init_rng (n_channels, n_x);
   //rng_generate (n_channels, n_x, n_events_per_channel, x);

   //printf ("Generated: \n");
   //for (int i = 0; i < d.n_events_gen; i++) {
   //   printf ("%lf\n", x[i]);
   //}

   //srand(1234);
   //for (int i = 0; i < n_x * d.n_events_gen; i++) {
   //   x[i] = (double)rand()/RAND_MAX;
   //}

   int *channels = (int*)malloc(d.n_events_gen * sizeof(int));
   for (int i = 0; i < d.n_events_gen; i++) {
     channels[i] = i / n_events_per_channel;
   }

   double *p = (double*)malloc(4 * n_out * d.n_events_gen * sizeof(double));
   double *factors = (double*)malloc(d.n_events_gen * sizeof(double)); 
   double *volumes = (double*)malloc(d.n_events_gen * sizeof(double)); 
   bool *oks_gpu = (bool*)malloc(d.n_events_gen * sizeof(bool));
   bool *oks_cpu = (bool*)malloc(d.n_events_gen * sizeof(bool));

   init_mapping_constants_cpu (n_channels, sqrts * sqrts, 0, sqrts * sqrts);
   init_phs_gpu(n_channels, mappings_host, sqrts * sqrts);
   //t1 = mysecond();
   //gen_phs_from_x_gpu (d, n_channels, channels, n_x, x, factors, volumes, oks, p);
   //t2 = mysecond();

   //printf ("GPU: %lf sec\n", t2 - t1);
   //int n_ok = 0;
   //for (int i = 0; i < d.n_events_gen; i++) {
   //  if (oks[i]) n_ok++;
   //}
   //printf ("Valid events: %d / %d\n", n_ok, d.n_events_gen);

   //update_weights (n_x, n_channels, d.n_events_gen, channels, x, oks);

   int n_ok;
   //for (int i = 0; i < n_trials; i++) {
   //   rng_generate (n_channels, n_x, n_trial_events, x);
   //   gen_phs_from_x_gpu (n_channels * n_trial_events, n_channels, channels,
   //                       n_x, x, factors, volumes, oks, p);
   //   n_ok = 0;
   //   for (int i = 0; i < d.n_events_gen; i++) {
   //     if (oks[i]) n_ok++;
   //   }
   //   printf ("Trial %d: %d / %d\n", i, n_ok, d.n_events_gen);

   //   update_weights (n_x, n_channels, d.n_events_gen, channels, x, oks);
   //}

   // Now do the real time measurement with the adapted grids
   rng_generate (n_channels, n_x, n_events_per_channel, x);
   t1 = mysecond();
   gen_phs_from_x_gpu (d.n_events_gen, n_channels, channels, n_x, x, factors, volumes, oks_gpu, p);
   t2 = mysecond();

   printf ("GPU: %lf sec\n", t2 - t1);
   n_ok = 0;
   for (int i = 0; i < d.n_events_gen; i++) {
     if (oks_gpu[i]) n_ok++;
   }
   printf ("Valid events: %d / %d\n", n_ok, d.n_events_gen);

   free(p);
   phs_prt_t *prt = (phs_prt_t*)malloc(N_PRT * d.n_events_gen * sizeof(phs_prt_t));
   memset (prt, 0, N_PRT * 4 * d.n_events_gen * sizeof(double));
   ///double *factors = (double*)malloc(d.n_events_gen * sizeof(double));
   ///double *volumes = (double*)malloc(d.n_events_gen * sizeof(double));
   //
   //for (int i = 0; i < d.n_events_gen; i++) {
   //   oks[i] = true;
   //   factors[i] = 1;
   //   volumes[i] = 1;
   //}

   t1 = mysecond();
   gen_phs_from_x_cpu (sqrts, d, n_x, x, channels, factors, volumes, oks_cpu, prt);
   t2 = mysecond();

   printf ("CPU: %lf sec\n", t2 - t1);

   n_ok = 0;
   for (int i = 0; i < d.n_events_gen; i++) {
     if (oks_cpu[i]) n_ok++;
   }
   printf ("Valid events: %d / %d\n", n_ok, d.n_events_gen);


   for (int i = 0; i < d.n_events_gen; i++) {
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
   fprintf (logfl[LOG_INPUT], "n_trees: %d\n", n_trees);
   fprintf (logfl[LOG_INPUT], "nx: %d\n", n_x);
   fprintf (logfl[LOG_INPUT], "NPRT: %d\n", N_PRT);
   fprintf (logfl[LOG_INPUT], "NPRT_OUT: %d\n", N_PRT_OUT);

   daughters1 = (int**)malloc(n_trees * sizeof(int*));
   daughters2 = (int**)malloc(n_trees * sizeof(int*));
   has_children = (int**)malloc(n_trees * sizeof(int*));
   mappings_host = (mapping_t*)malloc(n_trees * sizeof(mapping_t));
   for (int i = 0; i < n_trees; i++) {
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
   read_tree_structures (ref_file, n_trees, N_PRT, N_PRT_OUT, &filepos);

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
      do_verify_against_whizard (ref_file, n_x, n_trees, n_in, n_out, filepos);
   } else {
      do_verify_internal (1000, 10, 1000, n_x, n_trees, n_in, n_out);
   }

   final_logfiles();
   return 0;
}
