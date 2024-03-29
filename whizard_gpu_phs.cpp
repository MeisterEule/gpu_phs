#include <cstring>
#include <math.h>

#include "phs.h"
#include "global_phs.h"

extern "C" void c_whizard_set_particle_structure (int n_channels, int n_trees, int n_groves,
                                     int n_x, int n_in, int n_out) {
   N_EXT_IN = n_in;
   N_EXT_OUT = n_out;
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
}

extern "C" void c_whizard_init_mappings (int n_channels) {
   mappings_host = (mapping_t*)malloc(n_channels * sizeof(mapping_t));
   for (int c = 0; c < n_channels; c++) {
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
}

extern "C" void c_whizard_fill_mapping (int channel, int *map_ids, double *masses, double *widths) {
   for (int i = 0; i < N_PRT_OUT; i++) {
      mappings_host[channel].map_id[i] = map_ids[i]; 
      mappings_host[channel].masses[i] = masses[i];
      mappings_host[channel].widths[i] = widths[i];
   }
}
