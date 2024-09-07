#include <cstring>
#include <math.h>
#include <unistd.h>

#include "file_input.h"
#include "phs.h"
#include "global_phs.h"

int *channel_ids = NULL;

extern "C" void c_whizard_set_particle_structure (int *n_channels, int *n_trees, int *n_groves,
                                                  int *n_x, int *n_in, int *n_out) {
   N_CHANNELS = *n_channels;
   N_EXT_IN = *n_in;
   N_EXT_OUT = *n_out;
   N_EXT_TOT = N_EXT_IN + N_EXT_OUT;

   N_PRT = pow(2,N_EXT_TOT) - 1;
   N_PRT_OUT = pow(2, N_EXT_OUT) - 1;
   PRT_STRIDE = 4 * N_PRT;
   ROOT_BRANCH = N_PRT_OUT - 1;

   N_BRANCHES = 2*N_EXT_OUT - 1;
   N_BRANCHES_INTERNAL = N_BRANCHES - N_EXT_OUT;
   N_MSQ = N_EXT_OUT - 2;
   N_LAMBDA_IN = N_EXT_OUT - 1;
   N_BOOSTS = N_LAMBDA_IN + 1;
   N_LAMBDA_OUT = N_BRANCHES;

   flv_masses = (double*)malloc(N_EXT_TOT * sizeof(double));
   flv_widths = (double*)malloc(N_EXT_TOT * sizeof(double));
}

extern "C" void c_whizard_set_flavors (double *masses, double *widths) {
   for (int i = 0; i < N_EXT_TOT; i++) {
      flv_masses[i] = masses[i];
      flv_widths[i] = widths[i];
   }
}

extern "C" void c_whizard_init_mappings () {
   mappings_host = (mapping_t*)malloc(N_CHANNELS * sizeof(mapping_t));
   for (int c = 0; c < N_CHANNELS; c++) {
      mappings_host[c].map_id = (int*)malloc(N_PRT_OUT * sizeof(int));
      mappings_host[c].comp_msq = NULL;
      mappings_host[c].comp_ct = NULL;
      mappings_host[c].a = (map_constant_t*)malloc(N_PRT_OUT * sizeof(map_constant_t));
      mappings_host[c].b = (map_constant_t*)malloc(N_PRT_OUT * sizeof(map_constant_t));
      mappings_host[c].masses = (double*)malloc(N_PRT_OUT * sizeof(double));
      mappings_host[c].widths = (double*)malloc(N_PRT_OUT * sizeof(double));
      mappings_host[c].mass_sum = (double*)malloc(N_PRT_OUT * sizeof(double));
      memset (mappings_host[c].mass_sum, 0, N_PRT_OUT * sizeof(double));
   }
   ///fflush(stdout);
   ///sleep(1);
}

extern "C" void c_whizard_fill_mapping (int *channel, int *map_ids, double *masses, double *widths) {
   int c = *channel - 1; // Fortran -> C index
   for (int i = 0; i < N_PRT_OUT; i++) {
      mappings_host[c].map_id[i] = map_ids[i]; 
      mappings_host[c].masses[i] = masses[i];
      mappings_host[c].widths[i] = widths[i];
   }
}

extern "C" void c_whizard_init_tree_structures () {
   daughters1 = (int**)malloc(N_CHANNELS * sizeof(int*));
   daughters2 = (int**)malloc(N_CHANNELS * sizeof(int*));
   has_children = (int**)malloc(N_CHANNELS * sizeof(int*));
   contains_friends = (int*)malloc(N_CHANNELS * sizeof(int));
   for (int c = 0; c < N_CHANNELS; c++) {
      daughters1[c] = (int*)malloc(N_PRT * sizeof(int));
      daughters2[c] = (int*)malloc(N_PRT * sizeof(int));
      has_children[c] = (int*)malloc(N_PRT * sizeof(int));
   }
}

extern "C" void c_whizard_fill_tree_structure (int *channel, int *w_daughters1, int *w_daughters2, int *w_has_children, int *w_contains_friends) {
  int c = *channel - 1;
  for (int i = 0; i < N_PRT; i++) {
     daughters1[c][i] = w_daughters1[i];  
     daughters2[c][i] = w_daughters2[i];  
     has_children[c][i] = w_has_children[i];
  }
  contains_friends[c] = *w_contains_friends;
} 

extern "C" void c_whizard_init_channel_ids (int *batch_size, int *n_channels, int *channel_limits) {
   if (channel_ids != NULL) free (channel_ids);
   channel_ids = (int*)malloc(*batch_size * sizeof(int));
   int current_channel = 0;
   for (int i = 0; i < *batch_size; i++) {
      channel_ids[i] = current_channel;
      if (i + 1 == channel_limits[current_channel]) current_channel++;
   }
}

extern "C" void c_whizard_set_threads (int *msq_threads, int *cb_threads, int *ab_threads) {
   kernel_control.msq_threads = *msq_threads;
   kernel_control.cb_threads = *cb_threads;
   kernel_control.ab_threads = *ab_threads;
}

extern "C" void c_whizard_init_gpu_phs (double *sqrts) {
   init_phs_gpu (N_CHANNELS, mappings_host, *sqrts);
}

extern "C" void c_whizard_gen_phs_from_x_gpu (int *n_events, int *n_channels, int *n_x, double *x,
                                              double *factors, double *volumes, bool *oks,
                                              double *p, double *x_out) {
  input_control.do_inverse_mapping = true;
  gen_phs_from_x_gpu (true, (size_t)*n_events, *n_channels, channel_ids, *n_x, x,
                      factors, volumes, oks, p, x_out);
}

extern "C" void c_whizard_get_momentum_device_pointer (double *p) {
  ///p = p_transfer_to_whizard; 
}

extern "C" void c_whizard_show_module () {
   printf ("n_channels: %d\n", N_CHANNELS);
   printf ("n_ext_in: %d\n", N_EXT_IN);
   printf ("n_ext_out: %d\n", N_EXT_OUT);
   printf ("n_ext_tot: %d\n", N_EXT_TOT);
   printf ("n_prt: %d\n", N_PRT);
   printf ("n_prt_out: %d\n", N_PRT_OUT);
   printf ("stride: %d\n", PRT_STRIDE);
   printf ("root_branch: %d\n", ROOT_BRANCH);
   printf ("n_branches: %d\n", N_BRANCHES);
   printf ("n_branches_internal: %d\n", N_BRANCHES_INTERNAL);
   printf ("N_LAMBDA_IN: %d\n", N_LAMBDA_IN);
   printf ("N_LAMBDA_OUT: %d\n", N_LAMBDA_OUT);
   printf ("N_BOOSTS: %d\n", N_BOOSTS);
   printf ("\n");
   printf ("Mappings: \n");
   if (mappings_host == NULL) {
      printf (" MAPPINGS NOT SET UP!\n");
      return;
   }
   for (int c = 0; c < N_CHANNELS; c++) {
      printf (" c: %d\n", c);
      printf (" ids: ");
      for (int i = 0; i < N_PRT_OUT; i++) {
         printf ("%d ", mappings_host[c].map_id[i]);
      }
      printf ("\n");
      printf (" masses: ");
      for (int i = 0; i < N_PRT_OUT; i++) {
         printf ("%lf ", mappings_host[c].masses[i]);
      }
      printf ("\n");
      printf (" widths: ");
      for (int i = 0; i < N_PRT_OUT; i++) {
         printf ("%lf ", mappings_host[c].widths[i]);
      }
      printf ("\n");

      printf (" mass_sum: ");
      for (int i = 0; i < N_PRT; i++) {
         printf ("%lf ", mappings_host[c].mass_sum[i]);
      }
      printf ("\n");

   }

   printf ("Children: \n");
   for (int c = 0; c < N_CHANNELS; c++) {
      for (int i = 0; i < N_PRT; i++) {
         printf ("%d ", daughters1[c][i]);
      }
      printf ("\n");
      for (int i = 0; i < N_PRT; i++) {
         printf ("%d ", daughters2[c][i]);
      }
      printf ("\n");
      for (int i = 0; i < N_PRT; i++) {
         printf ("%d ", has_children[c][i]);
      }
      printf ("\n");
   }

   printf ("Recursion mappings: \n");
   printf ("i_gather: \n");
   for (int c = 0; c < N_CHANNELS; c++) {
      printf ("  channel %d: ", c);
      for (int i = 0; i < N_BRANCHES; i++) {
         fprintf (stdout, "%d ", i_gather[c][i]);
      } 
      fprintf (stdout, "\n");
   } 
   printf ("i_scatter: \n");
   for (int c = 0; c < N_CHANNELS; c++) {
      printf ("  channel %d: ", c);
      for (int i = 0; i < N_EXT_OUT; i++) {
         fprintf (stdout, "%d ", i_scatter[c][i]);
      } 
      fprintf (stdout, "\n");
   }
   printf ("cmd_msq: \n"); 
   for (int c = 0; c < N_CHANNELS; c++) {
      printf ("  channel %d: ", c);
      for (int cc = 0; cc < N_BRANCHES_INTERNAL; cc++) {
         fprintf (stdout, "%d %d -> %d\n", cmd_msq[3*N_BRANCHES_INTERNAL*c + 3*cc + 0],
                                     cmd_msq[3*N_BRANCHES_INTERNAL*c + 3*cc + 1],
                                     cmd_msq[3*N_BRANCHES_INTERNAL*c + 3*cc + 2]);
      }
      fprintf (stdout, "\n");
   }

   fprintf (stdout, "N_LAMBDA_IN: %d\n", N_LAMBDA_IN);
   fprintf (stdout, "boost_o:\n");
   for (int c = 0; c < N_CHANNELS; c++) {
      printf ("channel %d: ", c);
      for (int i = 0; i < N_LAMBDA_IN; i++) {
          printf ("%d %d %d\n", cmd_boost_o[3*N_LAMBDA_IN*c + 3*i], cmd_boost_o[3*N_LAMBDA_IN*c + 3*i + 1], cmd_boost_o[3*N_LAMBDA_IN*c + 3*i + 2]);
      }
   }

   fprintf (stdout, "N_LAMBDA_OUT: %d\n", N_LAMBDA_OUT);
   fprintf (stdout, "boost_t:\n");
   for (int c = 0; c < N_CHANNELS; c++) {
      printf ("channel %d: ", c);
      for (int i = 0; i < N_LAMBDA_OUT; i++) {
          printf ("%d %d\n", cmd_boost_t[2*N_LAMBDA_IN*c + 2*i], cmd_boost_t[2*N_LAMBDA_IN*c + 2*i + 1]);
      }
   }
   fprintf (stdout, "END OF MODULE\n");

}
