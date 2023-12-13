#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <math.h>

#include "monitoring.h"
#include "phs.h"
#include "mom_generator.h"

bool verify_against_whizard = true;

int *n_cmd = NULL;
int *cmd = NULL;

void do_verify_against_whizard (char *ref_file, int n_x, int n_trees, 
                                int n_in, int n_out, int filepos_start_mom) {
   phs_dim_t d;
   d.n_events_val = count_nevents_in_reference_file (ref_file, n_in + n_out, filepos_start_mom);

   fprintf (logfl[LOG_INPUT], "n_events in reference file: %d\n", d.n_events_val);

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


   fprintf (logfl[LOG_INPUT], "channel_limits: ");
   for (int i = 0; i < n_trees + 1; i++) {
      fprintf (logfl[LOG_INPUT], "%d ", channel_lims[i]);
   }
   fprintf (logfl[LOG_INPUT], "\n");
   fprintf (logfl[LOG_INPUT], "n_events to generate: %d\n", d.n_events_gen);

   long long mem_gpu = count_gpu_memory_requirements (d, n_x);

   double *p = (double*)malloc(PRT_STRIDE * d.n_events_gen * sizeof(double));
   double *factors = (double*)malloc(N_PRT * d.n_events_gen * sizeof(double)); 
   double *volumes = (double*)malloc(N_PRT * d.n_events_gen * sizeof(double)); 
   int *oks = (int*)malloc(N_PRT * d.n_events_gen * sizeof(int));

   init_mapping_constants_cpu (n_trees, sqrts * sqrts, 0, sqrts * sqrts);
   init_phs_gpu(n_trees, mappings_host, sqrts * sqrts);
   double t1 = mysecond();
   //gen_phs_from_x_gpu (sqrts, d, n_trees, channel_lims, n_x, x, factors, volumes, oks, p);
   gen_phs_from_x_gpu_2 (d, cmd, n_cmd[0], n_trees, channel_lims, n_x, x);
   double t2 = mysecond();
   return;

   double t_tot = 0;
   for (int i = 0; i < 5; i++) {
      t_tot += gpu_timers[i];
   }
   printf ("GPU Timers: \n");
   printf ("  Memcpy In: %lf s\n", gpu_timers[TIME_MEMCPY_IN]);
   printf ("  Memcpy Out: %lf s\n", gpu_timers[TIME_MEMCPY_OUT]);
   printf ("  Memcpy Boosts: %lf s\n", gpu_timers[TIME_MEMCPY_BOOST]);
   printf ("  Msq Kernels: %lf s\n", gpu_timers[TIME_KERNEL_MSQ]);
   printf ("  Ang Kernels: %lf s\n", gpu_timers[TIME_KERNEL_ANG]);
   printf ("  Total: %lf s\n", t_tot);

   FILE *fp = fopen ("compare.gpu", "w+");
   compare_phs_gpu_vs_ref (fp, d.n_events_val, d.n_events_gen,
                           channels, n_in, n_out, pval, p, factors, volumes);
   fclose(fp);
   fp = NULL;
   printf ("dt: %lf sec\n", t2 - t1);

   free (p);
   free (factors);
   free (volumes);
   free (oks);
  
   phs_prt_t *prt = (phs_prt_t*)malloc(N_PRT * d.n_events_gen * sizeof(phs_prt_t));
   factors = (double*)malloc(d.n_events_gen * sizeof(double));
   volumes = (double*)malloc(d.n_events_gen * sizeof(double));

   t1 = mysecond();
   gen_phs_from_x_cpu (sqrts, d, n_x, x, channels, factors, volumes, prt);
   t2 = mysecond();

   fp = fopen ("compare.cpu", "w+");
   compare_phs_cpu_vs_ref (fp, d.n_events_val, d.n_events_gen,
                           channels, n_in, n_out, pval, prt, factors, volumes);
   fclose(fp);
   fp = NULL;

   printf ("dt: %lf sec\n", t2 - t1);


   free (factors);
   free (volumes);
   free (prt);
   free (x);
   free (pval);
}

void do_verify_internal (int n_events_gen, int n_x, int n_trees) {
   double t1, t2;
   phs_dim_t d;
   d.n_events_gen = n_events_gen;
   d.n_events_val = n_events_gen;

   double sqrts = 1000;
   init_mapping_constants_cpu (n_trees, sqrts * sqrts, 0, sqrts * sqrts);

   double *x = (double*)malloc(n_x * d.n_events_gen * sizeof(double));

   srand(1234);
   /// GENERATE X
   phs_prt_t *prt = (phs_prt_t*)malloc(N_PRT * d.n_events_gen * sizeof(phs_prt_t));
   double *factors = (double*)malloc(d.n_events_gen * sizeof(double));
   double *volumes = (double*)malloc(d.n_events_gen * sizeof(double));

   t1 = mysecond();
   //gen_phs_from_x_cpu (sqrts, d, n_x, channels, factors, volumes, prt);
   t2 = mysecond();

   printf ("dt: %lf sec\n", t2 - t1);
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

   n_cmd = (int*)malloc(n_trees * sizeof(int));
   int n_tot = 0;
   for (int c = 0; c < n_trees; c++) {
      n_cmd[c] = 0;
      for (int i = 0; i < N_PRT_OUT; i++) {
         if (daughters1[c][i] > 0) n_cmd[c]++;
      }
      n_tot += n_cmd[c];
   }

   cmd = (int*)malloc(3 * n_tot * sizeof(int));
   //msq_cmd_t *cmd = (msq_cmd_t*)malloc(n_tot * sizeof(msq_cmd_t));

   for (int c = 0; c < n_trees; c++) {
      int cc = 0;
      for (int i = 0; i < N_PRT_OUT; i++) {
         if (daughters1[c][i] > 0) {
            cmd[3 * n_cmd[c] * c + 3 * cc + 0] = daughters1[c][i] - 1;
            cmd[3 * n_cmd[c] * c + 3 * cc + 1] = (i+1) - daughters1[c][i] - 1;
            cmd[3 * n_cmd[c] * c + 3 * cc + 2] = (i+1) - 1;
            cc++;
         } 
      }
   }

   for (int c = 0; c < n_trees; c++) {
      printf ("Channel: %d\n", c);
      for (int cc = 0; cc < n_cmd[c]; cc++) {
         printf ("%d %d -> %d\n", cmd[3 * n_cmd[c] * c + 3 * cc + 0],
                                  cmd[3 * n_cmd[c] * c + 3 * cc + 1],
                                  cmd[3 * n_cmd[c] * c + 3 * cc + 2]);
      }
   }
    
   if (verify_against_whizard) {
      do_verify_against_whizard (ref_file, n_x, n_trees, n_in, n_out, filepos);
   } else {
      //
   }

   final_logfiles();
   return 0;
}
