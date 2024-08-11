#include <vector>
#include <list>
#include <iostream>
#include <cstring>

#include "phs.h"
#include "global_phs.h"
#include "monitoring.h"
#include "file_input.h"

__device__ int DN_EXT_IN;
__device__ int DN_EXT_OUT;
__device__ int DN_EXT_TOT;

__device__ int DN_PRT;
__device__ int DN_PRT_OUT;
__device__ int DN_BRANCHES;
__device__ int DPRT_STRIDE;
__device__ int DPART_STRIDE;
__device__ int DN_BOOSTS;
__device__ int DN_LAMBDA_IN;
__device__ int DN_LAMBDA_OUT;

#include "mappings_gpu.hpp"

int **daughters1 = NULL;
int **daughters2 = NULL;
int **has_children = NULL;
int *contains_friends = NULL;

int **i_scatter = NULL;
int **i_gather = NULL;

int *cmd_msq = NULL;

int *cmd_boost_o = NULL;
int *cmd_boost_t = NULL;

template <typename T> void cudaMemcpyMaskedH2D (size_t N, int *idx, T *field_d, T *field_h) {
   T *tmp = (T*)malloc(N * sizeof(T));
   for (size_t i = 0; i < N; i++) {
      tmp[i] = field_h[idx[i]];
   }
   cudaMemcpy (field_d, tmp, N * sizeof(T), cudaMemcpyHostToDevice);
   free(tmp);
}

int search_in_igather (int c, int x) {
  for (int cc = 0; cc < N_BRANCHES; cc++) {
     if (i_gather[c][cc] == x) return cc;
  }
  return -1;
}

__global__ void _init_mappings (int n_channels, mapping_t *map_h) {
   mappings_d = (mapping_t*)malloc(n_channels * sizeof(mapping_t));
   for (int c = 0; c < n_channels; c++) {
      mappings_d[c].map_id = (int*)malloc(DN_BRANCHES * sizeof(int));
      mappings_d[c].comp_msq = (mapping_msq_sig**)malloc(DN_BRANCHES * sizeof(mapping_msq_sig*));
      mappings_d[c].comp_msq_inv = (mapping_msq_inv_sig**)malloc(DN_BRANCHES * sizeof(mapping_msq_sig*));
      mappings_d[c].comp_ct = (mapping_ct_sig**)malloc(DN_BRANCHES * sizeof(mapping_ct_sig*));
      mappings_d[c].comp_ct_inv = (mapping_ct_inv_sig**)malloc(DN_BRANCHES * sizeof(mapping_ct_sig*));
      mappings_d[c].a = (map_constant_t*)malloc(DN_BRANCHES * sizeof(map_constant_t));
      mappings_d[c].b = (map_constant_t*)malloc(DN_BRANCHES * sizeof(map_constant_t));
      mappings_d[c].masses = (double *)malloc(DN_BRANCHES * sizeof(double));
      mappings_d[c].widths = (double *)malloc(DN_BRANCHES * sizeof(double));
      for (int i = 0; i < DN_BRANCHES; i++) {
         mappings_d[c].comp_msq[i] = NULL;
         mappings_d[c].comp_msq_inv[i] = NULL;
         mappings_d[c].comp_ct[i] = NULL;
         mappings_d[c].comp_ct_inv[i] = NULL;
      }
      mappings_d[c].mass_sum = (double*)malloc(DN_BRANCHES * sizeof(double));
      memset (mappings_d[c].mass_sum, 0, DN_BRANCHES * sizeof(double));
   }
}

__global__ void _fill_mapids (int channel, int n_part, int *map_ids) {
   for (int i = 0; i < n_part; i++) {
      mappings_d[channel].map_id[i] = map_ids[i];
   }
}

__global__ void _fill_masses (int channel, int n_part, double *m, double *w, double *ms) {
   for (int i = 0; i < n_part; i++) {
      mappings_d[channel].masses[i] = m[i];
      mappings_d[channel].widths[i] = w[i];
      mappings_d[channel].mass_sum[i] = ms[i];
   }
}

void set_mappings (int channel) {
   for (int i = 0; i < N_BRANCHES; i++) {
      _set_mappings<<<1,1>>>(channel, i);
      cudaDeviceSynchronize();
   }
}

// A helper struct to cast one-dimensional arrays to 4 x 4 matrices, to use more intuitive
// matrix indices. This has no performance impact.
struct boost {
  double l[4][4];
};

__global__ void _init_boost (size_t N, double *L) {
   size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
   if (tid >= N) return;
   struct boost *LL = (struct boost*)(&L[16 * tid]);
   memset (LL->l, 0, 16 * sizeof(double));
   LL->l[0][0] = 1;
   LL->l[1][1] = 1;
   LL->l[2][2] = 1;
   LL->l[3][3] = 1;
}

__global__ void _init_x (xcounter_t *xc, double *x, size_t *id, int nx) {
   xc->nx = nx;
   xc->id_gpu = id;
   xc->id_cpu = 0;
   xc->x = x;
}

__global__ void _reset_x (xcounter_t *xc, int n_events) {
  memset (xc->id_gpu, 0, n_events * sizeof(double));
}

__global__ void _init_fv (size_t N, double *factors, double *volumes, bool *oks) {
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= N) return;
  for (size_t i = 0; i < DN_BRANCHES; i++) {
     factors[DN_BRANCHES * tid + i] = 1;
     volumes[DN_BRANCHES * tid + i] = 1;
  }
  oks[tid] = true;
}

__global__ void _init_f (size_t N, double *factors) {
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= N) return;
  for (size_t i = 0; i < DN_BRANCHES; i++) {
     factors[DN_BRANCHES * tid + i] = 1;
  }
}

__global__ void _set_device_constants (int _root_branch, int _n_in, int _n_out,
                                       int _n_prt, int _n_prt_out, int _prt_stride,
                                       int _n_branches, int _n_lambda_in, 
                                       int _n_lambda_out) {
  DN_EXT_IN = _n_in;
  DN_EXT_OUT = _n_out;
  DN_EXT_TOT = DN_EXT_IN + DN_EXT_OUT;
  DN_PRT = _n_prt;
  DN_PRT_OUT = _n_prt_out;
  DPRT_STRIDE = _prt_stride;
  DN_BRANCHES = _n_branches;
  DN_BOOSTS = _n_lambda_in + 1;
  DN_LAMBDA_IN = _n_lambda_in;
  DN_LAMBDA_OUT = _n_lambda_out;
  DPART_STRIDE = DN_BRANCHES * 4;
}

__global__ void _move_factors (size_t N, int *channels, int n_channels, double *local_factors, double *all_factors) {
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= N) return;
  int channel = channels[tid];
  all_factors[n_channels * tid + channel] = local_factors[DN_BRANCHES * tid];
}

__global__ void _move_factors (size_t N, int *all_channels, int this_channel, int n_channels, double *local_factors, double *all_factors) {
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= N) return;
  if (all_channels[tid] == this_channel) return;
  all_factors[n_channels * tid + this_channel] = local_factors[DN_BRANCHES * tid];
}

__global__ void _move_x (size_t n_events, int n_x, int *all_channels, double *local_x, double *all_x) {
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= n_events) return;
  int channel = all_channels[tid]; 
  for (int i = 0; i < n_x; i++) {
     all_x[n_x * n_events * channel + n_x * tid + i] = local_x[n_x * tid + i];
  }
}

__global__ void _move_x (size_t n_events, int n_x, int *all_channels, int this_channel, int n_channels, xcounter_t *xc, double *all_x) {
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= n_events) return;
  int original_channel = all_channels[tid];
  if (original_channel == this_channel) return;
  for (int i = 0; i < n_x; i++) {
     all_x[n_x * n_events * this_channel + n_x * tid + i] = xc->x[n_x * tid + i];
  }
}

void count_max_boosts (int *nboost_max, int *nboost, int branch_idx) {
   if (has_children[0][branch_idx]) {
      int k1 = daughters1[0][branch_idx];
      int k2 = daughters2[0][branch_idx];
      (*nboost)++;
      if (*nboost > *nboost_max) *nboost_max = *nboost;
      count_max_boosts (nboost_max, nboost, k1);
      count_max_boosts (nboost_max, nboost, k2);
      (*nboost)--;
   }
}

void extract_msq_branch_idx (std::vector<int> *cmd_list, int channel, int branch_idx) {
   int k1 = daughters1[channel][branch_idx];
   int k2 = daughters2[channel][branch_idx];
   if (has_children[channel][k1]) extract_msq_branch_idx (cmd_list, channel, k1);
   if (has_children[channel][k2]) extract_msq_branch_idx (cmd_list, channel, k2);
   cmd_list->push_back(branch_idx); 
}

typedef struct {
   int b[3];
} boost_cmd_t;

void extract_boost_origins (std::vector<boost_cmd_t> *cmd_list, int channel, int branch_idx, std::list<int> *parent_boost, int *boost_counter) {
   if (has_children[channel][branch_idx]) {
      int k1 = daughters1[channel][branch_idx];
      int k2 = daughters2[channel][branch_idx];
      boost_cmd_t b;
      b.b[0] = branch_idx == ROOT_BRANCH ? 0 : search_in_igather(channel, branch_idx);
      b.b[1] = *boost_counter;
      b.b[2] = parent_boost->back();
      cmd_list->push_back (b);
      parent_boost->push_back(*boost_counter);
      (*boost_counter)++; 
      extract_boost_origins (cmd_list, channel, k1, parent_boost, boost_counter); 
      extract_boost_origins (cmd_list, channel, k2, parent_boost, boost_counter); 
      parent_boost->pop_back();
   }
}

void extract_boost_targets (std::vector<boost_cmd_t> *cmd_list, int channel, int branch_idx, std::list<int> *boost_idx, int *boost_counter) {
   boost_cmd_t b;
   b.b[0] = boost_idx->back();
   b.b[1] = search_in_igather(channel, branch_idx);
   if (has_children[channel][branch_idx]) {
      b.b[2] = search_in_igather(channel, daughters1[channel][branch_idx]);
   } else {
      b.b[2] = -1;
   }
   cmd_list->push_back(b);
   if (has_children[channel][branch_idx]) {
      int k1 = daughters1[channel][branch_idx];
      int k2 = daughters2[channel][branch_idx];
      boost_idx->push_back(*boost_counter);
      (*boost_counter)++;
      extract_boost_targets (cmd_list, channel, k1, boost_idx, boost_counter); 
      extract_boost_targets (cmd_list, channel, k2, boost_idx, boost_counter); 
      boost_idx->pop_back();
   }
}

__global__ void _init_first_boost (size_t N, double *L) {
   size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
   if (tid >= N) return;
   struct boost *LL = (struct boost*)(&L[16 * DN_BOOSTS * tid]);
   memset (LL->l, 0, 16 * sizeof(double));
   LL->l[0][0] = 1;
   LL->l[1][1] = 1;
   LL->l[2][2] = 1;
   LL->l[3][3] = 1;
}

void init_phs_gpu (int n_channels, mapping_t *map_h, double sqrts) {

   _set_device_constants<<<1,1>>>(ROOT_BRANCH, N_EXT_IN, N_EXT_OUT,
                                  N_PRT, N_PRT_OUT, PRT_STRIDE, N_BRANCHES,
                                  N_LAMBDA_IN, N_LAMBDA_OUT);

   int **d1 = (int**)malloc(n_channels * sizeof(int*));
   int **d2 = (int**)malloc(n_channels * sizeof(int*));
   for (int c = 0; c < n_channels; c++) {
      d1[c] = (int*)malloc(N_BRANCHES_INTERNAL * sizeof(int));
      d2[c] = (int*)malloc(N_BRANCHES_INTERNAL * sizeof(int));
      int cc = 0;
      for (int i = 0; i < N_PRT && cc < N_BRANCHES_INTERNAL; i++) {
         if (daughters1[c][i] > 0) {
            d1[c][cc] = daughters1[c][i];
            d2[c][cc] = daughters2[c][i];
            cc++;
         }
      }
   }

   i_gather = (int**)malloc(n_channels * sizeof(int*));
   for (int c = 0; c < n_channels; c++) {
      i_gather[c] = (int*)malloc(N_BRANCHES * sizeof(int));
      i_gather[c][0] = ROOT_BRANCH;
      int i_part = 1;
      for (int i = 0; i < N_BRANCHES_INTERNAL; i++) {
         if (d1[c][i] > 0) {
           i_gather[c][i_part++] = d1[c][i] - 1;
           i_gather[c][i_part++] = d2[c][i] - 1;
         }
      }
      ///fprintf (logfl[LOG_INPUT], "i_gather[%d]: ", c);
      ///for (int i = 0; i < N_BRANCHES; i++) {
      ///   fprintf (logfl[LOG_INPUT], "%d ", i_gather[c][i]);
      ///} 
      ///fprintf (logfl[LOG_INPUT], "\n");
   }

   i_scatter = (int**)malloc(n_channels * sizeof(int*));
   for (int c = 0; c < n_channels; c++) {
      i_scatter[c] = (int*)malloc(N_EXT_OUT * sizeof(int));
      for (int i = 0; i < N_EXT_OUT; i++) {
         i_scatter[c][i] = -1;
         int idx = pow(2,i) - 1;
         for (int j = 0; j < N_BRANCHES; j++) {
            if (i_gather[c][j] == idx) {
               i_scatter[c][i] = j;
               break;
            }
         } 
      }
   }

   for (int c = 0; c < n_channels; c++) {
      ///fprintf (logfl[LOG_INPUT], "i_scatter[%d]: ", c);
      ///for (int i = 0; i < N_EXT_OUT; i++) {
      ///   fprintf (logfl[LOG_INPUT], "%d ", i_scatter[c][i]);
      ///} 
      ///fprintf (logfl[LOG_INPUT], "\n");
   }

   cmd_msq = (int*)malloc(3 * n_channels * N_BRANCHES_INTERNAL * sizeof(int));

   for (int c = 0; c < n_channels; c++) {
      for (int i = 0; i < N_PRT_OUT; i++) {
         daughters1[c][i]--;
         daughters2[c][i]--;
      }
   }

   std::vector<int> branch_idx_extract;
   for (int c = 0; c < n_channels; c++) {
     branch_idx_extract.clear();
     extract_msq_branch_idx (&branch_idx_extract, c, ROOT_BRANCH);
     for (int i = 0; i < branch_idx_extract.size(); i++) {
        int branch_idx = branch_idx_extract[i];
        for (int j = 0; j < N_BRANCHES_INTERNAL; j++) {
           int b1 = search_in_igather (c, d1[c][j] - 1);
           int b2 = search_in_igather (c, d2[c][j] - 1);
           if (i_gather[c][b1] + i_gather[c][b2] + 1 == branch_idx) {
              cmd_msq[3*N_BRANCHES_INTERNAL*c + 3*i + 0] = b1; 
              cmd_msq[3*N_BRANCHES_INTERNAL*c + 3*i + 1] = b2; 
              cmd_msq[3*N_BRANCHES_INTERNAL*c + 3*i + 2] = branch_idx == ROOT_BRANCH ? 0 : search_in_igather (c, branch_idx);
           }
        }
     }
   }


   for (int c = 0; c < n_channels; c++) {
      ///fprintf (logfl[LOG_INPUT], "Channel: %d\n", c);
      ///for (int cc = 0; cc < N_BRANCHES_INTERNAL; cc++) {
      ///   fprintf (logfl[LOG_INPUT], "%d %d -> %d\n", cmd_msq[3*N_BRANCHES_INTERNAL*c + 3*cc + 0],
      ///                               cmd_msq[3*N_BRANCHES_INTERNAL*c + 3*cc + 1],
      ///                               cmd_msq[3*N_BRANCHES_INTERNAL*c + 3*cc + 2]);
      ///}
   }

   cudaDeviceSynchronize();
   _init_mappings<<<1,1>>>(n_channels, map_h);
   int *tmp;
   cudaMalloc((void**)&tmp, N_BRANCHES * sizeof(int));
   for (int c = 0; c < n_channels; c++) {
       cudaMemcpyMaskedH2D<int> (N_BRANCHES, i_gather[c], tmp, map_h[c].map_id);
       _fill_mapids<<<1,1>>> (c, N_BRANCHES, tmp);
   }
   cudaFree(tmp);

   double *mass_sum = (double*)malloc(N_PRT * sizeof(double));
   double *m, *w, *ms;
   cudaMalloc((void**)&m, N_BRANCHES * sizeof(double));
   cudaMalloc((void**)&w, N_BRANCHES * sizeof(double));
   cudaMalloc((void**)&ms, N_BRANCHES * sizeof(double));
   for (int c = 0; c < n_channels; c++) {
       cudaMemcpyMaskedH2D<double> (N_BRANCHES, i_gather[c], m, map_h[c].masses);
       cudaMemcpyMaskedH2D<double> (N_BRANCHES, i_gather[c], w, map_h[c].widths);
       cudaMemcpyMaskedH2D<double> (N_BRANCHES, i_gather[c], ms, map_h[c].mass_sum);
       _fill_masses<<<1,1>>> (c, N_BRANCHES, m, w, ms);
   }

   cudaFree(m);
   cudaFree(w);
   cudaFree(ms);
   cudaDeviceSynchronize();
   _init_mapping_constants<<<1,1>>> (n_channels, N_BRANCHES, sqrts);
   for (int c = 0; c < n_channels; c++) {
      set_mappings(c);
   }

   std::vector<boost_cmd_t> cmd_origin;
   std::vector<boost_cmd_t> cmd_target;
   std::list<int> parent_stack;
   parent_stack.push_back(0);
   cmd_boost_o = (int*)malloc(3 * N_LAMBDA_IN * n_channels * sizeof(int));
   for (int c = 0; c < n_channels; c++) {
      cmd_origin.clear();
      int dummy = 1;
      extract_boost_origins (&cmd_origin, c, ROOT_BRANCH, &parent_stack, &dummy);
      if (cmd_origin.size() != N_LAMBDA_IN) {
         printf ("Internal error: Number of boosts does not equal stack size!\n");
      } else {
      }

      ///fprintf (logfl[LOG_INPUT], "N_LAMBDA_IN: %d, cmd_origin.size(): %ld\n", N_LAMBDA_IN, cmd_origin.size());
      ///fprintf (logfl[LOG_INPUT], "Origins[%d]\n", c);
      ///for (boost_cmd_t b : cmd_origin) {
      ///  fprintf (logfl[LOG_INPUT], "%d %d %d\n", b.b[0], b.b[1], b.b[2]);
      ///}


      for (int i = 0; i < N_LAMBDA_IN; i++) {
         cmd_boost_o[3*N_LAMBDA_IN*c + 3*i] = cmd_origin[i].b[0];
         cmd_boost_o[3*N_LAMBDA_IN*c + 3*i + 1] = cmd_origin[i].b[1];
         cmd_boost_o[3*N_LAMBDA_IN*c + 3*i + 2] = cmd_origin[i].b[2];
      }
  }

  // parent_stack should be at 0
  cmd_boost_t = (int*)malloc(3 * N_LAMBDA_OUT * n_channels * sizeof(int));
  for (int c = 0; c < n_channels; c++) {
     cmd_target.clear();
     int dummy = 1;
     extract_boost_targets (&cmd_target, c, ROOT_BRANCH, &parent_stack, &dummy);
    
     fprintf (logfl[LOG_INPUT], "N_LAMBDA_OUT: %d, cmd_target.size(): %ld\n", N_LAMBDA_OUT, cmd_target.size());
     fprintf (logfl[LOG_INPUT], "Targets[%d]\n", c);
     for (boost_cmd_t b : cmd_target) {
        fprintf (logfl[LOG_INPUT], "%d %d %d\n", b.b[0], b.b[1], b.b[2]);
      }

     for (int i = 0; i < N_LAMBDA_OUT; i++) {
        cmd_boost_t[3*N_LAMBDA_OUT*c + 3*i] = cmd_target[i].b[0];
        cmd_boost_t[3*N_LAMBDA_OUT*c + 3*i + 1] = cmd_target[i].b[1];
        cmd_boost_t[3*N_LAMBDA_OUT*c + 3*i + 2] = cmd_target[i].b[2];
     }
  }
  printf ("CudaError Init: %s\n", cudaGetErrorString(cudaGetLastError()));
}

__global__ void _init_msq (size_t N, int n_channels, int *channels,
                           int *i_gather, double *flv_masses, double *msq) {
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= N) return;
  int channel = channels[tid];
  for (int i = 0; i < DN_BRANCHES; i++) {
     int x = i_gather[DN_BRANCHES * channel + i] + 1;
     if ((x & (x - 1)) == 0) { // Power of 2
        int ld = 0;
        while (x > 1) {
           x = x / 2;
           ld++;
        }
        double m = flv_masses[DN_EXT_IN + ld];
        msq[DN_BRANCHES * tid + i] = m * m;
     }
  }
}

// This is the main kernel for the first step of momentum generation. Using the decay triplets
// stored in cmd, it calls the mapping function and fills the msq and p_decay arrays.
// The Root branch is treated separately, because there is no mapping function involved
// and factor and volume are just the products of the children variables.

__global__ void _apply_msq (size_t N, double sqrts, int *channels, int *cmd, int n_cmd,
                            xcounter_t *xc, double *p_decay,
                            double *msq, double *factors, double *volumes, bool *oks) {
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= N) return;
  int channel = channels[tid];
  double m_tot = mappings_d[channel].mass_sum[0];
  for (int c = 0; c < n_cmd - 1; c++) {
     int k1 = cmd[3 * n_cmd * channel + 3 * c];
     int k2 = cmd[3 * n_cmd * channel + 3 * c + 1];
     int branch_idx = cmd[3 * n_cmd * channel + 3 * c + 2]; 
     size_t xtid = xc->nx * tid + xc->id_gpu[tid]++;
     double x = xc->x[xtid];
     double *a = mappings_d[channel].a[branch_idx].a;
     double m = mappings_d[channel].masses[branch_idx];
     double w = mappings_d[channel].widths[branch_idx];
     double m_min = mappings_d[channel].mass_sum[branch_idx];
     double m_max = sqrts - m_tot + m_min;
     double f;
     mappings_d[channel].comp_msq[branch_idx](x, sqrts * sqrts, m, w, m_min*m_min, m_max*m_max, a,
                                              &msq[DN_BRANCHES * tid + branch_idx], &f);
     factors[DN_BRANCHES * tid + branch_idx] *= f * factors[DN_BRANCHES * tid + k1] * factors[DN_BRANCHES * tid + k2]; 
     volumes[DN_BRANCHES * tid + branch_idx] *= volumes[DN_BRANCHES * tid + k1] * volumes[DN_BRANCHES * tid + k2] * sqrts * sqrts / (4 * TWOPI2);

     double msq0 = msq[DN_BRANCHES * tid + branch_idx];
     double msq1 = msq[DN_BRANCHES * tid + k1];
     double msq2 = msq[DN_BRANCHES * tid + k2];
     double m0 = sqrt(msq0);
     double m1 = sqrt(msq1);
     double m2 = sqrt(msq2);
     double lda = (msq0 - msq1 - msq2) * (msq0 - msq1 - msq2) - 4 * msq1 * msq2; 
     p_decay[DN_BRANCHES * tid + k1] = sqrt(lda) / (2 * m0);
     p_decay[DN_BRANCHES * tid + k2] = -sqrt(lda) / (2 * m0);
     factors[DN_BRANCHES * tid + branch_idx] *= sqrt(lda) / msq0;
     oks[tid] &= (msq0 >= 0 && lda > 0 && m0 > m1 + m2 && m0 <= m_max);
  }

  // ROOT BRANCH
  int k1 = cmd[3 * n_cmd * channel + 3 * (n_cmd-1)];
  int k2 = cmd[3 * n_cmd * channel + 3 * (n_cmd-1) + 1];
  double m_max = sqrts;
  msq[DN_BRANCHES * tid] = sqrts * sqrts;
  factors[DN_BRANCHES * tid] = factors[DN_BRANCHES * tid + k1] * factors[DN_BRANCHES * tid + k2];
  volumes[DN_BRANCHES * tid] = volumes[DN_BRANCHES * tid + k1] * volumes[DN_BRANCHES * tid + k2] / (4 * TWOPI5);
  double msq0 = msq[DN_BRANCHES * tid];
  double msq1 = msq[DN_BRANCHES * tid + k1];
  double msq2 = msq[DN_BRANCHES * tid + k2];
  double m0 = sqrt(msq0);
  double m1 = sqrt(msq1);
  double m2 = sqrt(msq2);
  double lda = (msq0 - msq1 - msq2) * (msq0 - msq1 - msq2) - 4 * msq1 * msq2; 
  p_decay[DN_BRANCHES * tid + k1] = sqrt(lda) / (2 * m0);
  p_decay[DN_BRANCHES * tid + k2] = -sqrt(lda) / (2 * m0);
  factors[DN_BRANCHES * tid] *= sqrt(lda) / msq0;
  oks[tid] &= (msq0 >= 0 && lda > 0 && m0 > m1 + m2 && m0 <= m_max);
}

__global__ void _apply_msq_inv (size_t N, int channel, xcounter_t *xc, double *msq, double sqrts, int *channels, int *cmd, int n_cmd,
                                double *p_decay, double *factors) {
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= N) return;
  if (channel == channels[tid]) return;
  double m_tot = mappings_d[channel].mass_sum[0];
  for (int c = 0; c < n_cmd - 1; c++) {
     int k1 = cmd[3 * n_cmd * channel + 3 * c];
     int k2 = cmd[3 * n_cmd * channel + 3 * c + 1];
     int branch_idx = cmd[3 * n_cmd * channel + 3 * c + 2]; 
     double *a = mappings_d[channel].a[branch_idx].a;
     double m = mappings_d[channel].masses[branch_idx];
     double w = mappings_d[channel].widths[branch_idx];
     double m_min = mappings_d[channel].mass_sum[branch_idx];
     double m_max = sqrts - m_tot + m_min;
     double f;
     size_t xtid = xc->nx * tid + xc->id_gpu[tid]++;
     double *x = &(xc->x[xtid]);
     mappings_d[channel].comp_msq_inv[branch_idx](msq[DN_BRANCHES * tid + branch_idx], m_min*m_min, m_max*m_max, sqrts*sqrts,
                                                  m, w, a, x, &f);
     factors[DN_BRANCHES * tid + branch_idx] = f * factors[DN_BRANCHES * tid + k1] * factors[DN_BRANCHES * tid + k2];

     double msq0 = msq[DN_BRANCHES * tid + branch_idx];
     double msq1 = msq[DN_BRANCHES * tid + k1];
     double msq2 = msq[DN_BRANCHES * tid + k2];
     double m0 = sqrt(msq0);
     double m1 = sqrt(msq1);
     double m2 = sqrt(msq2);
     double lda = (msq0 - msq1 - msq2) * (msq0 - msq1 - msq2) - 4 * msq1 * msq2; 
     p_decay[DN_BRANCHES * tid + k1] = sqrt(lda) / (2 * m0);
     p_decay[DN_BRANCHES * tid + k2] = -sqrt(lda) / (2 * m0);
     factors[DN_BRANCHES * tid + branch_idx] *= sqrt(lda) / msq0;
  }

  // ROOT BRANCH
  int k1 = cmd[3 * n_cmd * channel + 3 * (n_cmd-1)];
  int k2 = cmd[3 * n_cmd * channel + 3 * (n_cmd-1) + 1];
  double m_max = sqrts;
  msq[DN_BRANCHES * tid] = sqrts * sqrts;
  factors[DN_BRANCHES * tid] = factors[DN_BRANCHES * tid + k1] * factors[DN_BRANCHES * tid + k2];
  double msq0 = msq[DN_BRANCHES * tid];
  double msq1 = msq[DN_BRANCHES * tid + k1];
  double msq2 = msq[DN_BRANCHES * tid + k2];
  double m0 = sqrt(msq0);
  double m1 = sqrt(msq1);
  double m2 = sqrt(msq2);
  double lda = (msq0 - msq1 - msq2) * (msq0 - msq1 - msq2) - 4 * msq1 * msq2; 
  p_decay[DN_BRANCHES * tid + k1] = sqrt(lda) / (2 * m0);
  p_decay[DN_BRANCHES * tid + k2] = -sqrt(lda) / (2 * m0);
  factors[DN_BRANCHES * tid] *= sqrt(lda) / msq0;
}

__global__ void _create_boosts (size_t N, double sqrts, int *channels, int *cmd, int n_cmd,
                                xcounter_t *xc, double *msq, double *p_decay,
                                double *Ld, double *factors) {
   size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
   if (tid >= N) return;
   int channel = channels[tid];
   for (int c = 0; c < n_cmd; c++) {
      int branch_idx = cmd[3*n_cmd*channel + 3*c];
      double p = p_decay[DN_BRANCHES * tid + branch_idx];
      double m = sqrt(msq[DN_BRANCHES * tid + branch_idx]);
      double bg = m > 0 ? p / m : 0;
      double gamma = sqrt (1 + bg * bg);

      size_t xtid = xc->nx * tid + xc->id_gpu[tid]++;
      double x = xc->x[xtid];
      double phi = x * TWOPI;
      double cp = cos(phi);
      double sp = sin(phi);

      double ct, st, f;
      xtid = xc->nx * tid + xc->id_gpu[tid]++;
      x = xc->x[xtid];
      double *b = mappings_d[channel].b[branch_idx].a;
      mappings_d[channel].comp_ct[branch_idx](x, sqrts * sqrts, b, &ct, &st, &f);
      factors[DN_BRANCHES * tid] *= f;

      double L1[4][4];
      L1[0][0] = gamma;
      L1[0][1] = -bg * st;
      L1[0][2] = 0;
      L1[0][3] = bg * ct;
      L1[1][0] = 0;
      L1[1][1] = ct * cp;
      L1[1][2] = -sp;
      L1[1][3] = st * cp;
      L1[2][0] = 0;
      L1[2][1] = ct * sp;
      L1[2][2] = cp;
      L1[2][3] = st * sp;
      L1[3][0] = bg;
      L1[3][1] = -gamma * st;
      L1[3][2] = 0;
      L1[3][3] = gamma * ct; 

      int parent_boost = cmd[3*n_cmd*channel + 3*c + 2];

      int boost_idx = cmd[3*n_cmd*channel + 3*c + 1];
      struct boost *L0 = (struct boost*)(&Ld[16 * DN_BOOSTS * tid + 16 * parent_boost]);
      struct boost *Lnew = (struct boost*)(&Ld[16 * DN_BOOSTS * tid + 16 * boost_idx]);
      memset (Lnew, 0, 16 * sizeof(double));
      for (int i = 0; i < 4; i++) {
         for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
               Lnew->l[i][j] += L0->l[i][k] * L1[k][j];
            }
         }
      }
   }
}

__device__ double azimuthal_angle (double n[3]) {
   double tmp = atan2(n[1], n[0]);
   return tmp < 0 ? tmp + TWOPI : tmp;
}

__device__ void polar_angle_ct (double n[3], double *ct, double *st) {
   double dn = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
   *ct = n[2] / dn;
   *st = sqrt(1 - (*ct)*(*ct));
}

__global__ void _create_boosts_inv (size_t N, double sqrts, int channel, int *channels, xcounter_t *xc, 
                                    double *phi, double *ct, double *st,
                                    int *cb_cmd, int n_cb_cmd, int *ab_cmd, int n_ab_cmd, double *msq, double *p_decay, double *prt,
                                    int *i_gather, double *Ld, double *factors) {
   size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
   if (tid >= N) return;
   if (channel == channels[tid]) return;

   int branch_idx = ab_cmd[3*n_ab_cmd*channel + 3*0 + 1];
   int daughter_idx = ab_cmd[3*n_ab_cmd*channel + 3*0 + 2];
   int prt_idx = i_gather[DN_BRANCHES * channel + daughter_idx];
   int boost_idx = cb_cmd[3*n_cb_cmd*channel + 3*0 + 1];
   int parent_boost = cb_cmd[3*n_cb_cmd*channel + 3*0 + 2];
   double m = sqrt(msq[DN_BRANCHES * tid + branch_idx]);
   double p = p_decay[DN_BRANCHES * tid + branch_idx];
   double bg = m > 0 ? p / m : 0;

   double n[3];
   n[0] = prt[DPRT_STRIDE * tid + 4 * prt_idx + 1];
   n[1] = prt[DPRT_STRIDE * tid + 4 * prt_idx + 2];
   n[2] = prt[DPRT_STRIDE * tid + 4 * prt_idx + 3];
   double gamma = sqrt(1 + bg * bg);
   n[2] = n[2] * gamma + prt[DPRT_STRIDE * tid + + 4 * prt_idx] * bg;
   
   phi[DN_BOOSTS * tid + boost_idx] = azimuthal_angle(n);
   size_t xtid = xc->nx * tid + xc->id_gpu[tid]++;
   double *x = &(xc->x[xtid]);
   *x = phi[DN_BOOSTS * tid + boost_idx] / TWOPI;
   polar_angle_ct(n, &ct[DN_BOOSTS * tid + boost_idx], &st[DN_BOOSTS * tid + boost_idx]);
   double f;
   double *b = mappings_d[channel].b[branch_idx].a;
   xtid = xc->nx * tid + xc->id_gpu[tid]++;
   x = &(xc->x[xtid]);
   mappings_d[channel].comp_ct_inv[branch_idx](ct[DN_BOOSTS * tid + boost_idx], st[DN_BOOSTS * tid + boost_idx], sqrts*sqrts, b, x, &f);
   factors[DN_BRANCHES * tid + branch_idx] *= f;

   struct boost *L1 = (struct boost*)(&Ld[16 * DN_BOOSTS * tid + 16 * boost_idx]);
   L1->l[0][0] = gamma; 
   L1->l[0][1] = 0;
   L1->l[0][2] = 0;
   L1->l[0][3] = -bg;
   L1->l[1][0] = 0;
   L1->l[1][1] = gamma;
   L1->l[1][2] = 0;
   L1->l[1][3] = 0;
   L1->l[2][0] = 0;
   L1->l[2][1] = 0;
   L1->l[2][2] = gamma;
   L1->l[2][3] = 0;
   L1->l[3][0] = -bg;
   L1->l[3][1] = 0;
   L1->l[3][2] = 0;
   L1->l[3][3] = gamma;

   for (int c = 1; c < n_cb_cmd; c++) {
      branch_idx = ab_cmd[3*n_ab_cmd*channel + 3*c + 1];
      daughter_idx = ab_cmd[3*n_ab_cmd*channel + 3*c + 2];
      prt_idx = i_gather[DN_BRANCHES * channel + daughter_idx];
      p = p_decay[DN_BRANCHES * tid + branch_idx];
      m = sqrt(msq[DN_BRANCHES * tid + branch_idx]);
      bg = m > 0 ? p / m : 0;
      gamma = sqrt (1 + bg * bg);

      parent_boost = cb_cmd[3*n_cb_cmd*channel + 3*c + 2];
      boost_idx = cb_cmd[3*n_cb_cmd*channel + 3*c + 1];
      double cp0 = cos(phi[DN_BOOSTS * tid + parent_boost]);
      double sp0 = sin(phi[DN_BOOSTS * tid + parent_boost]);
      double ct0 = ct[DN_BOOSTS * tid + parent_boost];
      double st0 = st[DN_BOOSTS * tid + parent_boost];
      double p1[4];
      struct boost *L0 = (struct boost*)(&Ld[16 * DN_BOOSTS * tid + 16 * parent_boost]);
      for (int i = 0; i < 4; i++) {
         p1[i] = 0;
         for (int j = 0; j < 4; j++) {
            p1[i] += prt[DPRT_STRIDE * tid + 4 * prt_idx + j] * L0->l[i][j];
         }
      }

     double px = cp0 * p1[1] + sp0 * p1[2];  
     double py = -sp0  * p1[1] + cp0 * p1[2];
     n[0] = ct0 * px - st0 * p1[3];
     n[1] = py;
     n[2] = -st0 * px + ct0 * p1[3];
     gamma = sqrt(1 + bg*bg);
     n[2] = n[2] * gamma - p1[0] * bg;
     phi[DN_BOOSTS * tid + boost_idx] = azimuthal_angle(n);
     polar_angle_ct (n, &ct[DN_BOOSTS * tid + boost_idx], &st[DN_BOOSTS * tid + boost_idx]);
     xtid = xc->nx * tid + xc->id_gpu[tid]++;
     x = &(xc->x[xtid]);
     mappings_d[channel].comp_ct_inv[branch_idx](ct[DN_BOOSTS * tid + boost_idx], st[DN_BOOSTS * tid + boost_idx], sqrts*sqrts, b, x, &f);
     factors[DN_BRANCHES * tid + branch_idx] *= f;

     double L1[4][4];
     L1[0][0] = gamma;
     L1[0][1] = -bg * st0 * cp0;
     L1[0][2] = -bg * st0 * sp0;
     L1[0][3] = -bg * ct0;
     L1[1][0] = 0;
     L1[1][1] = ct0 * cp0;
     L1[1][2] = ct0 * sp0;
     L1[1][3] = -st0;
     L1[2][0] = 0;
     L1[2][1] = -sp0;
     L1[2][2] = cp0;
     L1[2][3] = 0;
     L1[3][0] = -bg;
     L1[3][1] = gamma * st0 * cp0;
     L1[3][2] = gamma * st0 * sp0;
     L1[3][3] = gamma * ct0;

     struct boost *Lnew = (struct boost*)(&Ld[16 * DN_BOOSTS * tid + 16 * boost_idx]);

     for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
           for (int k = 0; k < 4; k++) {
              Lnew->l[i][j] += L0->l[i][k] * L1[k][j];
           }
        }
     }
   }
}                                  

__global__ void _apply_boost_targets (size_t N, int *channels, int *cmd, int n_cmd, int *i_gather,
                                      double *Ld, double *msq, double *p_decay, double *prt) {
   size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
   if (tid >= N) return;
   int channel = channels[tid];
   for (int c = 0; c < n_cmd; c++) {
      int boost_idx = cmd[3*n_cmd*channel + 3*c];
      int branch_idx = cmd[3*n_cmd*channel + 3*c + 1];
      int prt_idx = i_gather[DN_BRANCHES * channel + branch_idx];
      double p = p_decay[DN_BRANCHES * tid + branch_idx];
      double E = sqrt(msq[DN_BRANCHES * tid + branch_idx] + p * p);
      for (int i = 0; i < 4; i++) {
         prt[DPRT_STRIDE * tid + 4 * prt_idx + i] = Ld[16*DN_BOOSTS*tid + 16*boost_idx + 4*i] * E
                                                  + Ld[16*DN_BOOSTS*tid + 16*boost_idx + 4*i + 3] * p;
      }
   }
}

__global__ void _combine_particles (size_t N, int this_channel, int *channels, int *cmd, int n_cmd, int *i_gather, double *prt, double *msq) {
   size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
   if (tid >= N) return;
   if (this_channel == channels[tid]) return;
   for (int c = 0; c < n_cmd; c++) {
      int branch_idx = cmd[3*n_cmd*this_channel + 3*c + 2];
      int prt_idx = i_gather[DN_BRANCHES * this_channel + branch_idx];
      int k1 = cmd[3*n_cmd*this_channel + 3*c];
      k1 = i_gather[DN_BRANCHES * this_channel + k1];
      int k2 = cmd[3*n_cmd*this_channel + 3*c + 1];
      k2 = i_gather[DN_BRANCHES * this_channel + k2];
      for (int i = 0; i < 4; i++) {
         prt[DPRT_STRIDE * tid + 4 * prt_idx + i] = prt[DPRT_STRIDE * tid + 4 * k1 + i] + prt[DPRT_STRIDE * tid + 4 * k2 + i];
      }
   }
   for (int branch_idx = 0; branch_idx < DN_BRANCHES; branch_idx++) {
      int prt_idx = i_gather[DN_BRANCHES * this_channel + branch_idx];
      msq[DN_BRANCHES * tid + branch_idx] = prt[DPRT_STRIDE * tid + 4 * prt_idx] * prt[DPRT_STRIDE * tid + 4 * prt_idx]
                                          - prt[DPRT_STRIDE * tid + 4 * prt_idx + 1] * prt[DPRT_STRIDE * tid + 4 * prt_idx + 1]
                                          - prt[DPRT_STRIDE * tid + 4 * prt_idx + 2] * prt[DPRT_STRIDE * tid + 4 * prt_idx + 2]
                                          - prt[DPRT_STRIDE * tid + 4 * prt_idx + 3] * prt[DPRT_STRIDE * tid + 4 * prt_idx + 3];
   }
}

void gen_phs_from_x_gpu (bool for_whizard, size_t n_events, 
                         int n_channels, int *channels, int n_x, double *x_h,
                         double *factors_h, double *volumes_h, bool *oks_h,
                         double *p_h, double *x_out) {
   START_TIMER(TIME_MEMCPY_IN);
   double *x_d;
   size_t *id_d;
   cudaMalloc((void**)&x_d, n_x * n_events * sizeof(double));
   cudaMalloc((void**)&id_d, n_events * sizeof(size_t));
   xcounter_t *xc;
   cudaMalloc((void**)&xc, sizeof(xcounter_t));
   cudaMemcpy (x_d, x_h, n_x * n_events * sizeof(double), cudaMemcpyHostToDevice);
   cudaMemset (id_d, 0, n_events * sizeof(size_t));

   int *cmds_msq_d, *cmds_boost_o_d, *cmds_boost_t_d;
   cudaMalloc((void**)&cmds_msq_d, 3 * n_channels * N_BRANCHES_INTERNAL * sizeof(int));
   cudaMalloc((void**)&cmds_boost_o_d, 3 * n_channels * N_LAMBDA_IN * sizeof(int));
   cudaMalloc((void**)&cmds_boost_t_d, 3 * n_channels * N_LAMBDA_OUT * sizeof(int));
   cudaMemcpy(cmds_msq_d, cmd_msq, 3 * n_channels * N_BRANCHES_INTERNAL * sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(cmds_boost_o_d, cmd_boost_o, 3 * n_channels * N_LAMBDA_IN * sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(cmds_boost_t_d, cmd_boost_t, 3 * n_channels * N_LAMBDA_OUT * sizeof(int), cudaMemcpyHostToDevice);

   if (prt_d == NULL) cudaMalloc((void**)&prt_d, N_PRT * n_events * 4 * sizeof(double));
   cudaMemset(prt_d, 0, N_PRT * n_events * 4 * sizeof(double));

   double *msq_d;
   cudaMalloc((void**)&msq_d, N_BRANCHES * n_events * sizeof(double)); 
   cudaMemset(msq_d, 0, N_BRANCHES * n_events * sizeof(double));
   double *p_decay;
   cudaMalloc((void**)&p_decay, N_BRANCHES * n_events * sizeof(double));

   double *local_factors_d;
   cudaMalloc ((void**)&local_factors_d, N_BRANCHES * n_events * sizeof(double));
   double *all_factors_d = NULL;
   if (input_control.do_inverse_mapping) {
      cudaMalloc((void**)&all_factors_d, n_channels * n_events * sizeof(double));
   } else {
      ///cudaMalloc((void**)&all_factors_d, N_BRANCHES * n_events * sizeof(double));
   }
   double *all_x_d = NULL;
   if (input_control.do_inverse_mapping) cudaMalloc((void**)&all_x_d, n_channels * n_x * n_events * sizeof(double));
   double *volumes_d;
   cudaMalloc ((void**)&volumes_d, N_BRANCHES * n_events * sizeof(double));
   bool *oks_d;
   cudaMalloc ((void**)&oks_d, n_events * sizeof(bool));

   double *Ld;
   cudaMalloc((void**)&Ld, n_events * 16 * N_BOOSTS * sizeof(double));

   int *channels_d;
   cudaMalloc((void**)&channels_d, n_events * sizeof(int));
   cudaMemcpy (channels_d, channels, n_events * sizeof(int), cudaMemcpyHostToDevice);
   cudaDeviceSynchronize();
   STOP_TIMER(TIME_MEMCPY_IN);

   START_TIMER(TIME_KERNEL_INIT);
   _init_first_boost<<<n_events/1024 + 1,1024>>>(n_events, Ld); // Does not work with static __device__
   _init_fv<<<n_events/1024 + 1,1024>>> (n_events, local_factors_d, volumes_d, oks_d);
   _init_x<<<1,1>>> (xc, x_d, id_d, n_x);
   cudaDeviceSynchronize();
   STOP_TIMER(TIME_KERNEL_INIT);

   ///CHECK_CUDA_STATE(SAFE_CUDA_INIT);


   int nt = kernel_control.msq_threads;
   int nb = n_events / nt + 1;

   int *tmp, *i_gather_d;
   tmp = (int*)malloc(N_BRANCHES * n_channels * sizeof(int));
   for (int c = 0; c < n_channels; c++) {
      for (int i = 0; i < N_BRANCHES; i++) {
         tmp[N_BRANCHES * c + i] = i_gather[c][i];
      }
   }
   cudaMalloc((void**)&i_gather_d, n_channels * N_BRANCHES * sizeof(int));
   // Why does this not work? The array is probably not contiguous
   //cudaMemcpy (i_scatter_d, &i_scatter[0][0], n_channels * N_EXT_TOT * sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy (i_gather_d, tmp, n_channels * N_BRANCHES * sizeof(int), cudaMemcpyHostToDevice);

   free(tmp);
   double *flv_masses_d;
   cudaMalloc((void**)&flv_masses_d, N_EXT_TOT * sizeof(double));
   cudaMemcpy (flv_masses_d, flv_masses, N_EXT_TOT * sizeof(double), cudaMemcpyHostToDevice);
   _init_msq<<<nb,nt>>>(n_events, n_channels, channels_d, i_gather_d, flv_masses_d, msq_d);
   cudaDeviceSynchronize();

   double sqrts = 1000;
   START_TIMER(TIME_KERNEL_MSQ);
   // TODO: Filter sqrts < mass_sum. Irrelevant for fixed sqrts in lepton collisions.
   _apply_msq<<<nb,nt>>>(n_events, sqrts, channels_d, cmds_msq_d,
                         N_BRANCHES_INTERNAL, xc, p_decay, msq_d, local_factors_d, volumes_d, oks_d);
   cudaDeviceSynchronize();
   STOP_TIMER(TIME_KERNEL_MSQ);

   ///CHECK_CUDA_STATE(SAFE_CUDA_MSQ);

   nt = kernel_control.cb_threads;
   nb = n_events / nt  + 1;
   START_TIMER(TIME_KERNEL_CB);
   _create_boosts<<<nb,nt>>>(n_events, sqrts, channels_d, cmds_boost_o_d, N_LAMBDA_IN,
                             xc, msq_d, p_decay, Ld, local_factors_d);

   cudaDeviceSynchronize();
   STOP_TIMER(TIME_KERNEL_CB);
   ///CHECK_CUDA_STATE(SAFE_CUDA_CB);

   nt = kernel_control.ab_threads;
   nb = n_events / nt  + 1;
   START_TIMER(TIME_KERNEL_AB);
   _apply_boost_targets<<<nb,nt>>> (n_events, channels_d, cmds_boost_t_d, N_LAMBDA_OUT, i_gather_d,
                                    Ld, msq_d, p_decay, prt_d);

   cudaDeviceSynchronize();
   STOP_TIMER(TIME_KERNEL_AB);
   CHECK_CUDA_STATE(SAFE_CUDA_AB);

   if (input_control.do_inverse_mapping) {
      _move_factors<<<nb,nt>>> (n_events, channels_d, n_channels, local_factors_d, all_factors_d);
      _move_x<<<nb,nt>>> (n_events, n_x, channels_d, x_d, all_x_d);
      cudaDeviceSynchronize();
      double *phi_d, *ct_d, *st_d;
      cudaMalloc ((void**)&phi_d, N_BOOSTS * n_events * sizeof(double));
      cudaMemset (phi_d, 0, N_BOOSTS * n_events * sizeof(double));
      cudaMalloc ((void**)&ct_d, N_BOOSTS * n_events * sizeof(double));
      cudaMalloc ((void**)&st_d, N_BOOSTS * n_events * sizeof(double));
      for (int c = 0; c < n_channels; c++) {
         _reset_x<<<1,1>>> (xc, n_events);
         _combine_particles<<<nb,nt>>>(n_events, c, channels_d, cmds_msq_d, N_BRANCHES_INTERNAL, i_gather_d, prt_d, msq_d);
         _init_f<<<nb,nt>>>(n_events, local_factors_d);
         cudaDeviceSynchronize();
         _apply_msq_inv<<<nb,nt>>>(n_events, c, xc, msq_d, sqrts, channels_d, cmds_msq_d,
                                   N_BRANCHES_INTERNAL, p_decay, local_factors_d);
         cudaDeviceSynchronize();
         _create_boosts_inv<<<nb,nt>>> (n_events, sqrts, c, channels_d, xc,
                                        phi_d, ct_d, st_d,
                                        cmds_boost_o_d, N_LAMBDA_IN, cmds_boost_t_d, N_LAMBDA_OUT,
                                        msq_d, p_decay, prt_d, i_gather_d, Ld, local_factors_d);
         cudaDeviceSynchronize();
         _move_factors<<<nb,nt>>>(n_events, channels_d, c, n_channels, local_factors_d, all_factors_d);
         cudaDeviceSynchronize();
         _move_x<<<nb,nt>>>(n_events, n_x, channels_d, c, n_channels, xc, all_x_d);
         cudaDeviceSynchronize();
      }
      cudaFree(phi_d);
      cudaFree(ct_d);
      cudaFree(st_d);
   }

   START_TIMER(TIME_MEMCPY_OUT);
   // This can also be done on the device, primarily to avoid large temporary arrays.
   double *copy = (double*)malloc(4 * N_PRT * n_events * sizeof(double));
   cudaMemcpy (copy, prt_d, 4 * N_PRT * n_events * sizeof(double), cudaMemcpyDeviceToHost);
   for (size_t i = 0; i < n_events; i++) {
      int idx = 1;
      for (int j = 0; j < N_EXT_OUT; j++) {
         p_h[4*N_EXT_OUT*i + 4*j + 0] = copy[4*N_PRT*i + 4*(idx-1) + 0];
         p_h[4*N_EXT_OUT*i + 4*j + 1] = copy[4*N_PRT*i + 4*(idx-1) + 1];
         p_h[4*N_EXT_OUT*i + 4*j + 2] = copy[4*N_PRT*i + 4*(idx-1) + 2];
         p_h[4*N_EXT_OUT*i + 4*j + 3] = copy[4*N_PRT*i + 4*(idx-1) + 3];
         idx *= 2;
      }
   }

   free(copy);
   if (input_control.do_inverse_mapping) {
      copy = (double*)malloc(n_events * n_channels * sizeof(double));
      cudaMemcpy (copy, all_factors_d, n_channels * n_events * sizeof(double), cudaMemcpyDeviceToHost);
      for (int i = 0; i < n_events * n_channels; i++) {
         factors_h[i] = copy[i];
      }

      cudaMemcpy (x_out, all_x_d, n_events * n_x * n_channels * sizeof(double), cudaMemcpyDeviceToHost);
      
   } else {
      copy = (double*)malloc(n_events * N_BRANCHES * sizeof(double));
      cudaMemcpy (copy, local_factors_d, N_BRANCHES * n_events * sizeof(double), cudaMemcpyDeviceToHost);
      for (int i = 0; i < n_events; i++) {
         factors_h[i] = copy[N_BRANCHES*i];
      }
   }

   free(copy);
   copy = (double*)malloc(N_BRANCHES * n_events * sizeof(double));
   cudaMemcpy (copy, volumes_d, N_BRANCHES * n_events * sizeof(double), cudaMemcpyDeviceToHost);
   for (int i = 0; i < n_events; i++) {
      volumes_h[i] = copy[N_BRANCHES*i];
   }

   cudaMemcpy (oks_h, oks_d, n_events * sizeof(bool), cudaMemcpyDeviceToHost);
   STOP_TIMER(TIME_MEMCPY_OUT);

   free(copy);

   cudaFree(x_d);
   cudaFree(id_d);
   cudaFree(xc);
   cudaFree(cmds_msq_d);
   cudaFree(cmds_boost_o_d);
   cudaFree(cmds_boost_t_d);
   if (for_whizard) {
       n_events_in_store = n_events;
       //p_transfer_to_whizard = prt_d;  
       ///set_transfer_to_whizard<<<1,1>>>(prt_d);
   } else {
       cudaFree(prt_d);
   }
   cudaFree(msq_d);
   cudaFree(p_decay);
   cudaFree(local_factors_d);
   if (input_control.do_inverse_mapping) cudaFree(all_factors_d);
   cudaFree(volumes_d);
   cudaFree(oks_d);   
   cudaFree(Ld);
   cudaFree(channels_d);
   cudaFree(i_gather_d);
   cudaFree(flv_masses_d);
}

double *get_momentum_device_pointer () {
   return prt_d;
}
