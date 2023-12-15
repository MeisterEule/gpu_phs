#include <vector>
#include <list>
#include <iostream>
#include <cstring>

#include "phs.h"
#include "monitoring.h"

int N_PRT = 0;
int N_PRT_IN = 0;
int N_PRT_OUT = 0;
int PRT_STRIDE = 0;
int ROOT_BRANCH = 0;
//int N_PART = 0;
//int N_BOOSTS = 0;
int N_BRANCHES = 0;
int N_BRANCHES_INTERNAL = 0;
int N_MSQ = 0;
int N_BOOSTS = 0;
int N_LAMBDA_IN = 0;
int N_LAMBDA_OUT = 0; 


__device__ int DN_PRT;
__device__ int DN_PRT_OUT;
//__device__ int DN_PART;
__device__ int DN_BRANCHES;
__device__ int DPRT_STRIDE;
__device__ int DPART_STRIDE;
__device__ int DN_BOOSTS;
//__device__ int DROOT_BRANCH;
__device__ int DN_LAMBDA_IN;
__device__ int DN_LAMBDA_OUT;

#include "mappings_gpu.hpp"

int **daughters1 = NULL;
int **daughters2 = NULL;
int **has_children = NULL;

///int **boost_origins = NULL;
///int **boost_targets = NULL;

int **i_scatter = NULL;
int **i_gather = NULL;

//int *n_cmd_msq = NULL;
int *cmd_msq = NULL;

int *cmd_boost_o = NULL;
int *cmd_boost_t = NULL;
//int n_cmd_t = 0;

static double *m_max = NULL;

//__device__ double *Ld = NULL;

template <typename T> void cudaMemcpyMaskedH2D (int N, int *idx, T *field_d, T *field_h) {
   T *tmp = (T*)malloc(N * sizeof(T));
   for (int i = 0; i < N; i++) {
      tmp[i] = field_h[idx[i]];
   }
   cudaMemcpy (field_d, tmp, N * sizeof(T), cudaMemcpyHostToDevice);
   free(tmp);
}

int search_in_igather (int c, int x) {
  int idx;
  for (int cc = 0; cc < N_BRANCHES; cc++) {
     if (i_gather[c][cc] == x) return cc;
  }
  return -1;
}

__global__ void _init_mappings (int n_channels, int n_part, mapping_t *map_h) {
   mappings_d = (mapping_t*)malloc(n_channels * sizeof(mapping_t));
   for (int c = 0; c < n_channels; c++) {
      //mappings_d[c].map_id = (int*)malloc(DN_PRT_OUT * sizeof(int));
      mappings_d[c].map_id = (int*)malloc(n_part * sizeof(int));
      //mappings_d[c].comp_ct = (mapping_ct_sig**)malloc(DN_PRT_OUT * sizeof(mapping_ct_sig*));
      mappings_d[c].comp_ct = (mapping_ct_sig**)malloc(n_part * sizeof(mapping_ct_sig*));
      //mappings_d[c].comp_msq = (mapping_msq_sig**)malloc(DN_PRT_OUT * sizeof(mapping_msq_sig*));
      mappings_d[c].comp_msq = (mapping_msq_sig**)malloc(n_part * sizeof(mapping_msq_sig*));
      //mappings_d[c].a = (map_constant_t*)malloc(DN_PRT_OUT * sizeof(map_constant_t));
      //mappings_d[c].b = (map_constant_t*)malloc(DN_PRT_OUT * sizeof(map_constant_t));
      //mappings_d[c].masses = (double *)malloc(DN_PRT_OUT * sizeof(double));
      //mappings_d[c].widths = (double *)malloc(DN_PRT_OUT * sizeof(double));
      mappings_d[c].a = (map_constant_t*)malloc(n_part * sizeof(map_constant_t));
      mappings_d[c].b = (map_constant_t*)malloc(n_part * sizeof(map_constant_t));
      mappings_d[c].masses = (double *)malloc(n_part * sizeof(double));
      mappings_d[c].widths = (double *)malloc(n_part * sizeof(double));
      for (int i = 0; i < n_part; i++) {
         mappings_d[c].comp_ct[i] = NULL;
         mappings_d[c].comp_msq[i] = NULL;
      }
   }
}

__global__ void _fill_mapids (int channel, int n_part, int *map_ids) {
   for (int i = 0; i < n_part; i++) {
      mappings_d[channel].map_id[i] = map_ids[i];
   }
}

__global__ void _fill_masses (int channel, int n_part, double *m, double *w) {
   for (int i = 0; i < n_part; i++) {
      //printf ("MASS: %d %d %lf %lf\n", channel, i, m[i], w[i]);
      mappings_d[channel].masses[i] = m[i];
      mappings_d[channel].widths[i] = w[i];
   }
}

void set_mappings (int channel) {
   for (int i = 0; i < N_BRANCHES; i++) {
      _set_mappings<<<1,1>>>(channel, i);
      cudaDeviceSynchronize();
   }
}

__global__ void _set_msq_root (int N, int branch_idx, int k1, int k2, double *msq, double sqrts,
                               double *factors, double *volumes, double *m_max) {
   int tid = threadIdx.x + blockDim.x * blockIdx.x;
   if (tid >= N) return;
   msq[DN_PRT * tid + branch_idx] = sqrts * sqrts;
   m_max[DN_PRT * tid + branch_idx] = sqrts;  
   factors[DN_PRT * tid + branch_idx] = factors[DN_PRT * tid + k1] * factors[DN_PRT * tid + k2]; // get from mapping
   volumes[DN_PRT * tid + branch_idx] = volumes[DN_PRT * tid + k1] * volumes[DN_PRT * tid + k2] / (4 * TWOPI5); // get from mapping 
}

__global__ void _set_msq_branch (int N, int channel, int off, int branch_idx, int k1, int k2, xcounter_t *xc, double sqrts,
                                 double *msq, double *factors, double *volumes, int *oks, double *m_max) {
   int tid = threadIdx.x + blockDim.x * blockIdx.x;
   if (tid >= N) return;
   m_max[DN_PRT * tid + branch_idx] = sqrts;
   double msq_min = 0;
   double msq_max = sqrts * sqrts;
   double this_msq = 0;
   int gid = off + tid;
   int xtid = xc->nx * gid + xc->id_gpu[gid]++;
   double x = xc->x[xtid];
   double *a = mappings_d[channel].a[branch_idx].a;
   double m = mappings_d[channel].masses[branch_idx];
   double w = mappings_d[channel].widths[branch_idx];
   mappings_d[channel].comp_msq[branch_idx](x, sqrts * sqrts, m, w, msq_min, msq_max, a, &msq[DN_PRT * tid + branch_idx],
                                            &factors[DN_PRT * tid + branch_idx]);
   if (this_msq >= 0) {
     double m = sqrt(this_msq);
     factors[DN_PRT * tid + branch_idx] *= factors[DN_PRT * tid + k1] * factors[DN_PRT * tid + k2];
     volumes[DN_PRT * tid + branch_idx] *= volumes[DN_PRT * tid + k1] * volumes[DN_PRT * tid + k2] * sqrts * sqrts / (4 * TWOPI2);
   } else {
     oks[DN_PRT * tid + branch_idx] = 0;
   }
}

__global__ void _set_decay (int N, int branch_idx, int k1, int k2, int *oks, double *msq, double *m_max,
                            double *p_decay, double *factors) {
   int tid = threadIdx.x + blockDim.x * blockIdx.x;
   if (tid >= N) return;
   if (!oks[DN_PRT * tid + branch_idx]) return;
   double this_msq = msq[DN_PRT * tid + branch_idx];
   double msq1 = msq[DN_PRT * tid + k1];
   double msq2 = msq[DN_PRT * tid + k2];
   double m1 = sqrt(msq1);
   double m2 = sqrt(msq2);
   double m = sqrt(this_msq);
   double lda = (this_msq - msq1 - msq2) * (this_msq - msq1 - msq2) - 4 * msq1 * msq2;
   if (lda > 0 && m > m1 + m2 && m <= m_max[DN_PRT * tid + branch_idx]) {
      p_decay[DN_PRT * tid + k1] = sqrt(lda) / (2 * m);
      p_decay[DN_PRT * tid + k2] = -p_decay[DN_PRT * tid + k1]; 
      factors[DN_PRT * tid + branch_idx] *= sqrt(lda) / this_msq;
   } else {
      oks[DN_PRT * tid + branch_idx] = 0;
   }
}

void set_msq_gpu (phs_dim_t d, int channel, int off, int branch_idx, xcounter_t *xc, double sqrts, double *msq,
                  double *factors, double *volumes, int *oks, double *p_decay) {
   int k1 = daughters1[channel][branch_idx]; 
   int k2 = daughters2[channel][branch_idx];
   if (has_children[channel][k1]) {
      set_msq_gpu(d, channel, off, k1, xc, sqrts, msq, factors, volumes, oks, p_decay);
   } else {
      // fv are already initialized to 1
   }
   if (has_children[channel][k2]) {
      set_msq_gpu(d, channel, off, k2, xc, sqrts, msq, factors, volumes, oks, p_decay);
   } else {
      // fv are already initialized to 1
   }

   int nt = d.nt[channel];
   int nb = d.nb[channel];
   int batch = d.batch[channel];
   if (branch_idx == ROOT_BRANCH) {
      cudaMemset(m_max, 0, batch * N_PRT * sizeof(double));
      START_TIMER(TIME_KERNEL_MSQ);
      _set_msq_root<<<nb,nt>>>(batch, branch_idx, k1, k2, msq + N_PRT * off, sqrts, factors + N_PRT * off, volumes + N_PRT * off, m_max);
      cudaDeviceSynchronize();
      STOP_TIMER(TIME_KERNEL_MSQ);
   } else {
      START_TIMER(TIME_KERNEL_MSQ);
      _set_msq_branch<<<nb,nt>>>(batch, channel, off, branch_idx, k1, k2, xc, sqrts, msq + N_PRT * off,
                                     factors + N_PRT * off, volumes + N_PRT * off, oks + N_PRT * off, m_max);
      cudaDeviceSynchronize();
      STOP_TIMER(TIME_KERNEL_MSQ);
   }
   // if ok
   START_TIMER(TIME_KERNEL_MSQ);
   _set_decay<<<nb,nt>>> (batch, branch_idx, k1, k2, oks + N_PRT * off, msq + N_PRT * off, m_max, p_decay + N_PRT * off, factors + N_PRT * off);
   cudaDeviceSynchronize();
   STOP_TIMER(TIME_KERNEL_MSQ);
}

__global__ void _apply_boost (int N, int branch_idx, int *oks, double *p_decay, double *msq, double *prt, double *L) {
   int tid = threadIdx.x + blockDim.x * blockIdx.x;
   if (tid >= N) return;
   if (!oks[DN_PRT * tid + branch_idx]) return;
   double p = p_decay[DN_PRT * tid + branch_idx];
   double E = sqrt(msq[DN_PRT * tid + branch_idx] + p * p);
   for (int i = 0; i < 4; i++) {
      prt[DPRT_STRIDE * tid + 4 * branch_idx + i] = L[16 * tid + 4 * i] * E + L[16 * tid + 4 * i + 3] * p;
   }
}

struct boost {
  double l[4][4];
};

__global__ void _init_boost (int N, double *L) {
   int tid = threadIdx.x + blockDim.x * blockIdx.x;
   if (tid >= N) return;
   struct boost *LL = (struct boost*)(&L[16 * tid]);
   memset (LL->l, 0, 16 * sizeof(double));
   LL->l[0][0] = 1;
   LL->l[1][1] = 1;
   LL->l[2][2] = 1;
   LL->l[3][3] = 1;
}

__global__ void _create_new_boost (int N, int channel, int off, int branch_idx, xcounter_t *xc, int *oks, double s, double *p_decay,
                                   double *msq, double *factors, double *L0, double *L_new) {
   int tid = threadIdx.x + blockDim.x * blockIdx.x;
   if (tid >= N) return;
   if (!oks[DN_PRT * tid + branch_idx]) return;
   double p = p_decay[DN_PRT * tid + branch_idx];
   double m = sqrt(msq[DN_PRT * tid + branch_idx]);
   double bg = m > 0 ? p / m : 0;
   double gamma = sqrt (1 + bg * bg); 
   
   int gid = off + tid;
   int xtid = xc->nx * gid + xc->id_gpu[gid]++; 
   double x = xc->x[xtid];
   double phi = x * TWOPI;
   double cp = cos(phi);
   double sp = sin(phi);

   double ct, st, f;
   xtid = xc->nx * gid + xc->id_gpu[gid]++;
   x = xc->x[xtid];
   double *b = mappings_d[channel].b[branch_idx].a;
   mappings_d[channel].comp_ct[branch_idx](x, s, b, &ct, &st, &f);
   // For angle mappings, there are no interdependencies between branches.
   // Accumulate only on the root branch
   factors[DN_PRT * tid] *= f;

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

   struct boost *LL_0 = (struct boost*)(&L0[16 * tid]);
   struct boost *LL_new = (struct boost*)(&L_new[16 * tid]);
   memset (LL_new, 0, 16 * sizeof(double));
   for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
         for (int k = 0; k < 4; k++) {
            LL_new->l[i][j] += LL_0->l[i][k] * L1[k][j];
         }
      }
   }
}

void set_angles_gpu (phs_dim_t d, int channel, int off, int branch_idx, xcounter_t *xc, int *oks, double s, double *msq, double *factors,
                     double *p_decay, double *prt, double *L0) {
   int nt = d.nt[channel];
   int nb = d.nb[channel];
   int batch = d.batch[channel];
   double *L0_copy;
   cudaMalloc((void**)&L0_copy, 16 * batch * sizeof(double));
   START_TIMER (TIME_MEMCPY_BOOST);
   cudaMemcpy(L0_copy, L0, 16 * batch * sizeof(double), cudaMemcpyDeviceToDevice);
   STOP_TIMER (TIME_MEMCPY_BOOST);
   START_TIMER (TIME_KERNEL_ANG);
   _apply_boost<<<nb,nt>>>(batch, branch_idx, oks + N_PRT * off, p_decay + N_PRT * off,
                               msq + N_PRT * off, prt + PRT_STRIDE * off, L0_copy);
   cudaDeviceSynchronize();
   STOP_TIMER (TIME_KERNEL_ANG);
   if (has_children[channel][branch_idx]) {
      int k1 = daughters1[channel][branch_idx];
      int k2 = daughters2[channel][branch_idx];
      double *L_new;
      cudaMalloc((void**)&L_new, 16 * batch * sizeof(double));
      cudaMemset (L_new, 0, 16 * batch * sizeof(double));
      START_TIMER (TIME_KERNEL_ANG);
      _create_new_boost<<<nb,nt>>>(batch, channel, off, branch_idx, xc, oks + N_PRT * off, s, p_decay + N_PRT * off,
                                       msq + N_PRT * off, factors + N_PRT * off, L0_copy, L_new);
      cudaDeviceSynchronize();
      STOP_TIMER (TIME_KERNEL_ANG);

      set_angles_gpu (d, channel, off, k1, xc, oks, s, msq, factors, p_decay, prt, L_new);
      set_angles_gpu (d, channel, off, k2, xc, oks, s, msq, factors, p_decay, prt, L_new);
      cudaFree (L_new);
   }
   cudaFree (L0_copy); 
}


__global__ void _init_x (xcounter_t *xc, double *x, int *id, int nx) {
   xc->nx = nx;
   xc->id_gpu = id;
   xc->id_cpu = 0;
   xc->x = x;
}

__global__ void _init_fv (int N, double *factors, double *volumes, int *oks) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= N) return;
  for (int i = 0; i < DN_PRT; i++) {
     factors[DN_PRT * tid + i] = 1;
     volumes[DN_PRT * tid + i] = 1;
     oks[DN_PRT * tid + i] = 1;
  }
}

__global__ void _set_device_constants (int _n_prt, int _n_prt_out, int _prt_stride,
                                       int _n_branches, int _n_lambda_in,
                                       int _n_lambda_out, int _root_branch) {
  DN_PRT = _n_prt;
  DN_PRT_OUT = _n_prt_out;
  DPRT_STRIDE = _prt_stride;
  //DN_BOOSTS = _n_boosts + 1;
  //DROOT_BRANCH = _root_branch;
  DN_BRANCHES = _n_branches;
  DN_BOOSTS = _n_lambda_in + 1;
  DN_LAMBDA_IN = _n_lambda_in;
  DN_LAMBDA_OUT = _n_lambda_out;
  DPART_STRIDE = DN_BRANCHES * 4;
}

void count_max_boosts (int *nboost_max, int *nboost, int branch_idx) {
   //(*nboost)++;
   //if (*nboost > *nboost_max) *nboost_max = *nboost;
   if (has_children[0][branch_idx]) {
      int k1 = daughters1[0][branch_idx];
      int k2 = daughters2[0][branch_idx];
      (*nboost)++;
      if (*nboost > *nboost_max) *nboost_max = *nboost;
      count_max_boosts (nboost_max, nboost, k1);
      count_max_boosts (nboost_max, nboost, k2);
      (*nboost)--;
   }
   //(*nboost)--;
}

typedef struct {
   int b[3];
} boost_cmd_t;

void extract_boost_origins (std::vector<boost_cmd_t> *cmd_list, int channel, int branch_idx, std::list<int> *parent_boost, int *boost_counter) {
   if (has_children[channel][branch_idx]) {
      int k1 = daughters1[channel][branch_idx];
      int k2 = daughters2[channel][branch_idx];
      boost_cmd_t b;
      //b.b[0] = branch_idx;
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
   //b.b[1] = branch_idx;
   b.b[1] = search_in_igather(channel, branch_idx);
   b.b[2] = -1;
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

long long count_gpu_memory_requirements (phs_dim_t d, int n_x) {
#define BYTES_PER_GB 1073741824
   long long mem_m_max = N_PRT * d.n_events_gen * sizeof(double);
   long long mem_xc = n_x * d.n_events_gen * (sizeof(double) + sizeof(int));
   long long mem_prt = PRT_STRIDE * d.n_events_gen * sizeof(double);
   long long mem_pdecay = N_PRT * d.n_events_gen * sizeof(double);
   long long mem_msq = N_PRT * d.n_events_gen * sizeof(double);
   long long mem_factors = N_PRT * d.n_events_gen * sizeof(double);
   long long mem_volumes = N_PRT * d.n_events_gen * sizeof(double);
   long long mem_oks = N_PRT * d.n_events_gen * sizeof(double);

   int n_boost_max = 0;
   int dummy = 0;
   count_max_boosts (&n_boost_max, &dummy, ROOT_BRANCH);
   // Initial boost
   n_boost_max++;;
   long long mem_boost = n_boost_max * d.n_events_gen * 16 * sizeof(double); 

   long long mem_tot = mem_m_max + mem_xc + mem_prt 
                     + mem_pdecay + mem_msq + mem_factors
                     + mem_volumes + mem_oks + mem_boost;

   fprintf (logfl[LOG_CUDA], "GPU Memory requirements [GiB]: \n");
   fprintf (logfl[LOG_CUDA], "   m_max: %lf\n", (double)mem_m_max / BYTES_PER_GB);
   fprintf (logfl[LOG_CUDA], "   x_counter: %lf\n", (double)mem_xc / BYTES_PER_GB);
   fprintf (logfl[LOG_CUDA], "   momenta tree: %lf\n", (double)mem_prt / BYTES_PER_GB);
   fprintf (logfl[LOG_CUDA], "   p_decay: %lf\n", (double)mem_pdecay / BYTES_PER_GB);
   fprintf (logfl[LOG_CUDA], "   msq: %lf\n", (double)mem_msq / BYTES_PER_GB);
   fprintf (logfl[LOG_CUDA], "   factors: %lf\n", (double)mem_factors / BYTES_PER_GB);
   fprintf (logfl[LOG_CUDA], "   volumes: %lf\n", (double)mem_volumes / BYTES_PER_GB);
   fprintf (logfl[LOG_CUDA], "   oks: %lf\n", (double)mem_oks / BYTES_PER_GB);
   fprintf (logfl[LOG_CUDA], "   Boosts: %lf\n", (double)mem_boost / BYTES_PER_GB);
   fprintf (logfl[LOG_CUDA], "    Total: %lf\n", (double)mem_tot / BYTES_PER_GB);
   return mem_tot;
}

__global__ void _init_first_boost (int N, double *L) {
   int tid = threadIdx.x + blockDim.x * blockIdx.x;
   if (tid >= N) return;
   struct boost *LL = (struct boost*)(&L[16 * DN_BOOSTS * tid]);
   memset (LL->l, 0, 16 * sizeof(double));
   LL->l[0][0] = 1;
   LL->l[1][1] = 1;
   LL->l[2][2] = 1;
   LL->l[3][3] = 1;
  
}

void init_phs_gpu (int n_channels, mapping_t *map_h, double s) {

   //n_cmd_msq = (int*)malloc(n_channels * sizeof(int));
   //int n_tot = 0;
   //for (int c = 0; c < n_channels; c++) {
   //   n_cmd_msq[c] = 0;
   //   for (int i = 0; i < N_PRT; i++) {
   //      if (daughters1[c][i] > 0) n_cmd_msq[c]++;
   //   }
   //   n_tot += n_cmd_msq[c];
   //}

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
      printf ("i_gather%d: ", c);
      for (int i = 0; i < N_BRANCHES; i++) {
         printf ("%d ", i_gather[c][i]);
      } 
      printf ("\n");
   }


   //i_scatter = (int**)malloc(n_channels * sizeof(int*));
   //for (int c = 0; c < n_channels; c++) {
   //   i_scatter[c] = (int*)malloc(N_PRT * sizeof(int));
   //   memset(i_scatter[c], 0, N_PRT * sizeof(int));
   //   i_scatter[c][ROOT_BRANCH] = 0;
   //   int cc = 1;
   //   for (int i = 0; i < ROOT_BRANCH; i++) {
   //      if (daughters1[c][i] > 0) i_scatter[c][i] = cc++;
   //   }
   //}

   cmd_msq = (int*)malloc(3 * n_channels * N_BRANCHES_INTERNAL * sizeof(int));
   //msq_cmd_t *cmd = (msq_cmd_t*)malloc(n_tot * sizeof(msq_cmd_t));

   for (int c = 0; c < n_channels; c++) {
      for (int i = 0; i < N_BRANCHES_INTERNAL; i++) {
         // Find index of daughter in i_gather
         int b1, b2, b3;
         b1 = search_in_igather (c, d1[c][i] - 1);
         b2 = search_in_igather (c, d2[c][i] - 1);
                
         int tmp = i_gather[c][b1] + i_gather[c][b2] + 1;
         if (tmp == ROOT_BRANCH) {
            b3 = 0;         
         } else {
            b3 = search_in_igather (c, tmp);
         }
         cmd_msq[3*N_BRANCHES_INTERNAL*c + 3*i + 0] = b1;
         cmd_msq[3*N_BRANCHES_INTERNAL*c + 3*i + 1] = b2;
         cmd_msq[3*N_BRANCHES_INTERNAL*c + 3*i + 2] = b3;
      }
   }

   for (int c = 0; c < n_channels; c++) {
      printf ("Channel: %d\n", c);
      for (int cc = 0; cc < N_BRANCHES_INTERNAL; cc++) {
         ///printf ("%d %d -> %d\n", cmd[3 * n_cmd[c] * c + 3 * cc + 0] + 1,
         ///                         cmd[3 * n_cmd[c] * c + 3 * cc + 1] + 1,
         ///                         cmd[3 * n_cmd[c] * c + 3 * cc + 2] + 1);
         printf ("%d %d -> %d\n", cmd_msq[3*N_BRANCHES_INTERNAL*c + 3*cc + 0],
                                  cmd_msq[3*N_BRANCHES_INTERNAL*c + 3*cc + 1],
                                  cmd_msq[3*N_BRANCHES_INTERNAL*c + 3*cc + 2]);
      }
   }

   for (int c = 0; c < n_channels; c++) {
      for (int i = 0; i < N_PRT_OUT; i++) {
         daughters1[c][i]--;
         daughters2[c][i]--;
      }
   }


   cudaDeviceSynchronize();
   _init_mappings<<<1,1>>>(n_channels, N_BRANCHES, map_h);
   int *tmp;
   cudaMalloc((void**)&tmp, N_BRANCHES * sizeof(int));
   for (int c = 0; c < n_channels; c++) {
       //cudaMemcpy (tmp, map_h[c].map_id, N_PRT_OUT * sizeof(int), cudaMemcpyHostToDevice);
       cudaMemcpyMaskedH2D<int> (N_BRANCHES, i_gather[c], tmp, map_h[c].map_id);
       _fill_mapids<<<1,1>>> (c, N_BRANCHES, tmp);
   }
   cudaFree(tmp);

   double *m, *w;
   cudaMalloc((void**)&m, N_BRANCHES * sizeof(double));
   cudaMalloc((void**)&w, N_BRANCHES * sizeof(double));
   for (int c = 0; c < n_channels; c++) {
       //printf ("%lf %lf %lf %lf %lf\n", map_h[c].masses[i_gather[c][0]],
       //                                 map_h[c].masses[i_gather[c][1]],
       //                                 map_h[c].masses[i_gather[c][2]],
       //                                 map_h[c].masses[i_gather[c][3]],
       //                                 map_h[c].masses[i_gather[c][4]]);
       cudaMemcpyMaskedH2D<double> (N_BRANCHES, i_gather[c], m, map_h[c].masses);
       cudaMemcpyMaskedH2D<double> (N_BRANCHES, i_gather[c], w, map_h[c].widths);
       _fill_masses<<<1,1>>> (c, N_BRANCHES, m, w);
   }
   cudaFree(m);
   cudaFree(w);
   cudaDeviceSynchronize();
   _init_mapping_constants<<<1,1>>> (n_channels, N_BRANCHES, s, 0, s);
   for (int c = 0; c < n_channels; c++) {
      set_mappings(c);
   }

   //int dummy = 0;
   //int n_boosts = 0;
   //count_max_boosts (&n_boosts, &dummy, ROOT_BRANCH);
   //printf ("Max Boosts: %d\n", n_boosts);

   //boost_origins = (int**)malloc(n_channels * sizeof(int*));
   //boost_targets = (int**)malloc(n_channels * sizeof(int*));
   int _n_boosts;
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

      printf ("LOOP CMD:\n");
      fflush(stdout);

      for (boost_cmd_t b : cmd_origin) {
        std::cout << b.b[0] << " " << b.b[1] << " " << b.b[2] << "\n";
      }


      //boost_origins[c] = (int*)malloc(3 * _n_boosts * sizeof(int));
      for (int i = 0; i < N_LAMBDA_IN; i++) {
         cmd_boost_o[3*N_LAMBDA_IN*c + 3*i] = cmd_origin[i].b[0];
         cmd_boost_o[3*N_LAMBDA_IN*c + 3*i + 1] = cmd_origin[i].b[1];
         cmd_boost_o[3*N_LAMBDA_IN*c + 3*i + 2] = cmd_origin[i].b[2];
         //boost_origins[c][3*i] = cmd_origin[i].b[0];
         //boost_origins[c][3*i+1] = cmd_origin[i].b[1];
         //boost_origins[c][3*i+2] = cmd_origin[i].b[2];
      }
  }
  //N_BOOSTS = _n_boosts;

  // parent_stack should be at 0
  cmd_boost_t = (int*)malloc(2 * N_LAMBDA_OUT * n_channels * sizeof(int));
  for (int c = 0; c < n_channels; c++) {
     cmd_target.clear();
     int dummy = 1;
     extract_boost_targets (&cmd_target, c, ROOT_BRANCH, &parent_stack, &dummy);
     //n_cmd_t = cmd_target.size();
     //boost_targets[c] = (int*)malloc(2 * _n_boosts * sizeof(int));
     printf ("N_LAMBDA_OUT: %d, cmd_target.size(): %d\n", N_LAMBDA_OUT, cmd_target.size());
     printf ("Targets: %d\n", c);
     for (boost_cmd_t b : cmd_target) {
        std::cout << b.b[0] << " " << b.b[1] << "\n";
      }

     for (int i = 0; i < N_LAMBDA_OUT; i++) {
        //boost_targets[c][2*i] = cmd_target[i].b[0];
        //boost_targets[c][2*i+1] = cmd_target[i].b[1];
        cmd_boost_t[2*N_LAMBDA_OUT*c + 2*i] = cmd_target[i].b[0];
        cmd_boost_t[2*N_LAMBDA_OUT*c + 2*i + 1] = cmd_target[i].b[1];
     }
  }

  //printf ("N_BOOSTS: %d\n", N_BOOSTS);
  printf ("N_TARGETS: %d\n", N_LAMBDA_OUT);
  _set_device_constants<<<1,1>>>(N_PRT, N_PRT_OUT, PRT_STRIDE, N_BRANCHES,
                                 N_LAMBDA_IN, N_LAMBDA_OUT, ROOT_BRANCH);
}



void gen_phs_from_x_gpu_batch (double sqrts, phs_dim_t d, int n_channels, int *channel_offsets, 
                               int n_x, double *x_h, double *factors_h, double *volumes_h, int *oks_h, double *p_h) {

   cudaMalloc((void**)&m_max, N_PRT * d.n_events_gen * sizeof(double));

   double *x_d;
   int *id_d;
   cudaMalloc((void**)&x_d, n_x * d.n_events_gen * sizeof(double));
   cudaMalloc((void**)&id_d, d.n_events_gen * sizeof(int));
   xcounter_t *xc;
   cudaMalloc((void**)&xc, sizeof(xcounter_t));
   START_TIMER(TIME_MEMCPY_IN);
   cudaMemcpy (x_d, x_h, n_x * d.n_events_gen * sizeof(double), cudaMemcpyHostToDevice);
   STOP_TIMER(TIME_MEMCPY_IN);
   cudaMemset (id_d, 0, d.n_events_gen * sizeof(int));
   _init_x<<<1,1>>> (xc, x_d, id_d, n_x);

   memset (p_h, 0, PRT_STRIDE * d.n_events_gen * sizeof(double));
   double *p_d;
   cudaMalloc((void**)&p_d, PRT_STRIDE * d.n_events_gen * sizeof(double));
   cudaMemset(p_d, 0, PRT_STRIDE * d.n_events_gen * sizeof(double));

   double *pdecay_d;
   cudaMalloc((void**)&pdecay_d, N_PRT * d.n_events_gen * sizeof(double));
   cudaMemset (pdecay_d, 0, N_PRT * d.n_events_gen * sizeof(double));

   double *msq_d;
   cudaMalloc((void**)&msq_d, N_PRT * d.n_events_gen * sizeof(double));
   cudaMemset (msq_d, 0, N_PRT * d.n_events_gen * sizeof(double));

   double *factors_d;
   cudaMalloc ((void**)&factors_d, N_PRT * d.n_events_gen * sizeof(double));

   double *volumes_d;
   cudaMalloc ((void**)&volumes_d, N_PRT * d.n_events_gen * sizeof(double));

   int *oks_d;
   cudaMalloc ((void**)&oks_d, N_PRT * d.n_events_gen * sizeof(int));
   _init_fv<<<d.n_events_gen/1024 + 1,1024>>> (d.n_events_gen, factors_d, volumes_d, oks_d);
   cudaDeviceSynchronize();
   d.batch = (int*)malloc(n_channels * sizeof(int));
   d.nt = (int*)malloc(n_channels * sizeof(int));
   d.nb = (int*)malloc(n_channels * sizeof(int));
   for (int c = 0; c < n_channels; c++) {
      d.batch[c] = channel_offsets[c+1] - channel_offsets[c];
      if (d.batch[c] > 1024) {
         d.nt[c] = 1024;
         d.nb[c] = d.batch[c] / 1024 + 1; 
      } else {
         d.nt[c] = d.batch[c];
         d.nb[c] = 1;
      }
   }
      
   for (int channel = 0; channel < n_channels; channel++) {
      int off = channel_offsets[channel];
      set_msq_gpu (d, channel, off, ROOT_BRANCH, xc, sqrts,
                   msq_d, factors_d, volumes_d, oks_d, pdecay_d);
      fprintf (logfl[LOG_CUDA], "CUDA error msq %d: %s\n", channel, cudaGetErrorString(cudaGetLastError()));
   }

   for (int channel = 0; channel < n_channels; channel++) {
      int off = channel_offsets[channel];

      double *L0;
      cudaMalloc((void**)&L0, 16 * d.batch[channel] * sizeof(double));
      _init_boost<<<d.nb[channel],d.nt[channel]>>>(d.batch[channel], L0);
      set_angles_gpu (d, channel, off, N_PRT_OUT - 1, xc, oks_d, sqrts * sqrts,
                      msq_d, factors_d, pdecay_d, p_d, L0);
      cudaFree(L0);
      fprintf (logfl[LOG_CUDA], "CUDA error angles %d: %s\n", channel, cudaGetErrorString(cudaGetLastError()));
   }


   START_TIMER (TIME_MEMCPY_OUT);
   cudaMemcpy (p_h, p_d, PRT_STRIDE * d.n_events_gen * sizeof(double), cudaMemcpyDeviceToHost);
   cudaMemcpy (factors_h, factors_d, N_PRT * d.n_events_gen * sizeof(double), cudaMemcpyDeviceToHost);
   cudaMemcpy (volumes_h, volumes_d, N_PRT * d.n_events_gen * sizeof(double), cudaMemcpyDeviceToHost);
   cudaMemcpy (oks_h, oks_d, N_PRT * d.n_events_gen * sizeof(int), cudaMemcpyDeviceToHost);
   STOP_TIMER (TIME_MEMCPY_OUT);

   cudaFree (m_max);
   cudaFree (x_d);
   cudaFree (id_d);
   cudaFree (xc);
   cudaFree (p_d);
   cudaFree (pdecay_d);
   cudaFree (msq_d);
   cudaFree (factors_d);
   cudaFree (volumes_d);
   cudaFree (oks_d);

   free (d.batch);
   free (d.nt);
   free (d.nb);
}

__global__ void _apply_msq (int N, double sqrts, int channel, int *cmd, int n_cmd,
                            xcounter_t *xc,
                            double *msq, double *factors, double *volumes) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= N) return;
  double mm_max = sqrts;
  double msq_min = 0;
  double msq_max = sqrts * sqrts;
  for (int c = 0; c < n_cmd - 1; c++) {
     int k1 = cmd[3 * n_cmd * channel + 3 * c];
     int k2 = cmd[3 * n_cmd * channel + 3 * c + 1];
     int branch_idx = cmd[3 * n_cmd * channel + 3 * c + 2]; 
     int xtid = xc->nx * tid + xc->id_gpu[tid]++;
     double x = xc->x[xtid];
     if (tid == 0) printf ("c: %d, x: %lf\n", c, x);
     double *a = mappings_d[channel].a[branch_idx].a;
     double m = mappings_d[channel].masses[branch_idx];
     double w = mappings_d[channel].widths[branch_idx];
     if (tid == 0) printf ("c: %d, m: %lf, w: %lf\n", c, m, w);
     double f;
     if (tid == 0) printf ("c: %d, branch_idx: %d, map_id: %d\n",
                            c, branch_idx, mappings_d[channel].map_id[branch_idx]);
     mappings_d[channel].comp_msq[branch_idx](x, sqrts * sqrts, m, w, msq_min, msq_max, a,
                                              &msq[DN_BRANCHES * tid + branch_idx], &f);
     if (tid == 0) printf ("Mapping computed: %d %lf\n", branch_idx, msq[DN_BRANCHES * tid + branch_idx]);
     factors[DN_BRANCHES * tid + branch_idx] = f * factors[DN_BRANCHES * tid + k1] * factors[DN_BRANCHES * tid + k2]; 
     volumes[DN_BRANCHES * tid + branch_idx] *= volumes[DN_BRANCHES * tid + k1] * volumes[DN_BRANCHES * tid + k2] * sqrts * sqrts / (4 * TWOPI2);
  }
  // ROOT BRANCH
  int k1 = cmd[3 * n_cmd * channel + 3 * (n_cmd-1)];
  int k2 = cmd[3 * n_cmd * channel + 3 * (n_cmd-1) + 1];
  msq[DN_BRANCHES * tid] = sqrts * sqrts;
  factors[DN_BRANCHES * tid] = factors[DN_BRANCHES * tid + k1] * volumes[DN_BRANCHES * tid + k2] / (4 * TWOPI5);
}

__global__ void _apply_decay (int N, double sqrts, int channel, int *cmd, int n_cmd,
                              double *msq, double *p_decay, double *factors) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= N) return;
  for (int c = 0; c < n_cmd; c++) {
     int k1 = cmd[3 * n_cmd * channel + 3 * c];
     int k2 = cmd[3 * n_cmd * channel + 3 * c + 1];
     int branch_idx = cmd[3 * n_cmd * channel + 3 * c + 2]; 
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
}

__global__ void _create_boosts (int N, double sqrts, int channel, int *cmd, int n_cmd,
                                xcounter_t *xc, double *msq, double *p_decay,
                                double *Ld, double *factors) {
   int tid = threadIdx.x + blockDim.x * blockIdx.x;
   if (tid >= N) return;
   if (tid == 0) printf ("N_CMD: %d\n", n_cmd);
   for (int c = 0; c < n_cmd; c++) {
      int branch_idx = cmd[3*n_cmd*channel + 3*c];
      double p = p_decay[DN_BRANCHES * tid + branch_idx];
      double m = sqrt(msq[DN_BRANCHES * tid + branch_idx]);
      double bg = m > 0 ? p / m : 0;
      double gamma = sqrt (1 + bg * bg);

      //if (channel == 0 && tid == 0) printf ("id_gpu: %d\n", xc->id_gpu[tid]);
      int xtid = xc->nx * tid + xc->id_gpu[tid]++;
      double x = xc->x[xtid];
      //if (channel == 0 && tid == 0) printf ("xphi: %lf\n", x);
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
      if (tid == 0) printf ("boost_idx: %d %d\n", 3*n_cmd * c + 1, boost_idx);
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

__global__ void _apply_boost_targets (int N, int channel, int *cmd, int n_cmd,
                                      double *Ld, double *msq, double *p_decay, double *prt) {
   int tid = threadIdx.x + blockDim.x * blockIdx.x;
   if (tid >= N) return;
   for (int c = 0; c < n_cmd; c++) {
      int boost_idx = cmd[2*n_cmd*channel + 2*c];
      int branch_idx = cmd[2*n_cmd*channel + 2*c + 1];
      double p = p_decay[DN_BRANCHES * tid + branch_idx];
      double E = sqrt(msq[DN_BRANCHES * tid + branch_idx] + p * p);
      if (tid == 0) {
           printf ("p: %lf, E: %lf\n", p, E);
           printf ("Apply boost: %d -> %d\n", boost_idx, branch_idx);
           printf ("%lf %lf\n", Ld[16*DN_BOOSTS*tid + 16*boost_idx + 4*0],
                                Ld[16*DN_BOOSTS*tid + 16*boost_idx + 4*0 + 3]);
           printf ("%lf %lf\n", Ld[16*DN_BOOSTS*tid + 16*boost_idx + 4*1],
                                Ld[16*DN_BOOSTS*tid + 16*boost_idx + 4*1 + 3]);
           printf ("%lf %lf\n", Ld[16*DN_BOOSTS*tid + 16*boost_idx + 4*2],
                                Ld[16*DN_BOOSTS*tid + 16*boost_idx + 4*2 + 3]);
           printf ("%lf %lf\n", Ld[16*DN_BOOSTS*tid + 16*boost_idx + 4*3],
                                Ld[16*DN_BOOSTS*tid + 16*boost_idx + 4*3 + 3]);
      }
      for (int i = 0; i < 4; i++) {
         prt[DPART_STRIDE * tid + 4 * branch_idx + i] = Ld[16*DN_BOOSTS*tid + 16*boost_idx + 4*i] * E
                                                      + Ld[16*DN_BOOSTS*tid + 16*boost_idx + 4*i + 3] * p;
      }
      if (tid == 0) {
         printf ("prt: %d -> %d\n", boost_idx, branch_idx);
         printf ("%lf %lf %lf %lf\n", prt[DPART_STRIDE * tid + 4 * branch_idx], 
                                      prt[DPART_STRIDE * tid + 4 * branch_idx + 1],
                                      prt[DPART_STRIDE * tid + 4 * branch_idx + 2],
                                      prt[DPART_STRIDE * tid + 4 * branch_idx + 3]);
      }
   }
}

void gen_phs_from_x_gpu_2 (phs_dim_t d, 
                           int n_channels, int *channel_lims, int n_x, double *x_h) {
   double *x_d;
   int *id_d;
   cudaMalloc((void**)&x_d, n_x * d.n_events_gen * sizeof(double));
   cudaMalloc((void**)&id_d, d.n_events_gen * sizeof(int));
   xcounter_t *xc;
   cudaMalloc((void**)&xc, sizeof(xcounter_t));
   cudaMemcpy (x_d, x_h, n_x * d.n_events_gen * sizeof(double), cudaMemcpyHostToDevice);
   cudaMemset (id_d, 0, d.n_events_gen * sizeof(int));
   _init_x<<<1,1>>> (xc, x_d, id_d, n_x);

   int *cmds_msq_d, *cmds_boost_o_d, *cmds_boost_t_d;
   cudaMalloc((void**)&cmds_msq_d, 3 * n_channels * N_BRANCHES_INTERNAL * sizeof(int));
   cudaMalloc((void**)&cmds_boost_o_d, 3 * n_channels * N_LAMBDA_IN * sizeof(int));
   cudaMalloc((void**)&cmds_boost_t_d, 2 * n_channels * N_LAMBDA_OUT * sizeof(int));
   cudaMemcpy(cmds_msq_d, cmd_msq, 3 * n_channels * N_BRANCHES_INTERNAL * sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(cmds_boost_o_d, cmd_boost_o, 3 * n_channels * N_LAMBDA_IN * sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(cmds_boost_t_d, cmd_boost_t, 2 * n_channels * N_LAMBDA_OUT * sizeof(int), cudaMemcpyHostToDevice);

   double *prt_d;
   cudaMalloc((void**)&prt_d, N_BRANCHES * d.n_events_gen * 4 * sizeof(double));
   cudaMemset (prt_d, 0, N_BRANCHES * d.n_events_gen * 4 * sizeof(double));

   double *msq_d;
   cudaMalloc((void**)&msq_d, N_BRANCHES * d.n_events_gen * sizeof(double)); 
   cudaMemset(msq_d, 0, N_BRANCHES * d.n_events_gen * sizeof(double));
   double *p_decay;
   cudaMalloc((void**)&p_decay, N_BRANCHES * d.n_events_gen * sizeof(double));

   double *factors_d;
   cudaMalloc ((void**)&factors_d, N_PRT * d.n_events_gen * sizeof(double));
   double *volumes_d;
   cudaMalloc ((void**)&volumes_d, N_PRT * d.n_events_gen * sizeof(double));
   int *oks_d;
   cudaMalloc ((void**)&oks_d, N_PRT * d.n_events_gen * sizeof(int));
   _init_fv<<<d.n_events_gen/1024 + 1,1024>>> (d.n_events_gen, factors_d, volumes_d, oks_d);
   printf ("Init: %s\n", cudaGetErrorString(cudaGetLastError()));

   double *Ld;
   cudaMalloc((void**)&Ld, d.n_events_gen * 16 * N_BOOSTS * sizeof(double));
   _init_first_boost<<<d.n_events_gen/1024 + 1,1024>>>(d.n_events_gen, Ld); // Does not work with static __device__
   cudaDeviceSynchronize();
   printf ("Init: %s\n", cudaGetErrorString(cudaGetLastError()));

   d.batch = (int*)malloc(n_channels * sizeof(int));
   d.nt = (int*)malloc(n_channels * sizeof(int));
   d.nb = (int*)malloc(n_channels * sizeof(int));
   for (int c = 0; c < n_channels; c++) {
      d.batch[c] = channel_lims[c+1] - channel_lims[c];
      if (d.batch[c] > 1024) {
         d.nt[c] = 1024;
         d.nb[c] = d.batch[c] / 1024 + 1; 
      } else {
         d.nt[c] = d.batch[c];
         d.nb[c] = 1;
      }
   }

   cudaDeviceSynchronize();

   double sqrts = 1000;
   for (int channel = 0; channel < n_channels; channel++) {
      int nt = d.nt[channel];
      int nb = d.nb[channel];
      _apply_msq<<<nb,nt>>>(d.batch[channel], sqrts, channel, cmds_msq_d,
                            N_BRANCHES_INTERNAL, xc, msq_d, factors_d, volumes_d);
      cudaDeviceSynchronize();
      double *foo = (double*)malloc(N_BRANCHES * d.batch[channel] * sizeof(double));
      cudaMemcpy (foo, msq_d, N_BRANCHES * d.batch[channel] * sizeof(double), cudaMemcpyDeviceToHost);
   
      if (channel == 0) {
         printf ("msq: ");
         for (int i = 0; i < N_BRANCHES; i++) {
            printf ("%lf ", foo[i]); 
         }
         printf ("\n");
      }
      _apply_decay<<<nb,nt>>>(d.batch[channel], sqrts, channel, cmds_msq_d,
                              N_BRANCHES_INTERNAL, msq_d, p_decay, factors_d); 
      cudaDeviceSynchronize();
      printf ("MSQ %d: %s\n", channel, cudaGetErrorString(cudaGetLastError()));
      cudaMemcpy (foo, p_decay, N_BRANCHES * d.batch[channel] * sizeof(double), cudaMemcpyDeviceToHost);
   
      if (channel == 0) {
         printf ("decay: ");
         for (int i = 0; i < N_BRANCHES; i++) {
            printf ("%lf ", foo[i]); 
         }
         printf ("\n");
      }

      //_create_boosts
      _create_boosts<<<nb,nt>>>(d.batch[channel], sqrts, channel, cmds_boost_o_d, N_LAMBDA_IN,
                                xc, msq_d, p_decay, Ld, factors_d);
      cudaDeviceSynchronize();
      printf ("Create boost: %s\n", cudaGetErrorString(cudaGetLastError()));
      free(foo);
      //int nbo = N_BOOSTS + 1;
      foo = (double*)malloc(N_BOOSTS * 16 * sizeof(double));
      cudaMemcpy(foo, Ld, N_BOOSTS * 16 * sizeof(double), cudaMemcpyDeviceToHost);
      if (channel == 0) {
         for (int b = 0; b < N_BOOSTS; b++) {
            for (int i = 0; i < 4; i++) {
               printf ("%lf %lf %lf %lf\n", foo[16*b + 4*i], foo[16*b + 4*i+1],
                                            foo[16*b + 4*i+2], foo[16*b + 4*i+3]);
            }
         }
      }
      _apply_boost_targets<<<nb,nt>>> (d.batch[channel], channel, cmds_boost_t_d, N_LAMBDA_OUT,
      //_apply_boost_targets<<<1,1>>> (d.batch[channel], channel, cmds_boost_t_d, N_LAMBDA_OUT,
                                     Ld, msq_d, p_decay, prt_d);

      cudaDeviceSynchronize();
      printf ("Apply boost: %s\n", cudaGetErrorString(cudaGetLastError()));
      free(foo);
      foo = (double*)malloc(N_BRANCHES * 4 * d.batch[channel] * sizeof(double));
      cudaMemcpy(foo, prt_d, N_BRANCHES * 4 * d.batch[channel] * sizeof(double), cudaMemcpyDeviceToHost);
      if (channel == 0) {
         printf ("%lf %lf %lf %lf\n", foo[0], foo[1], foo[2], foo[3]);
         printf ("%lf %lf %lf %lf\n", foo[4], foo[5], foo[6], foo[7]);
         printf ("%lf %lf %lf %lf\n", foo[8], foo[9], foo[10], foo[11]);
         printf ("%lf %lf %lf %lf\n", foo[12], foo[13], foo[14], foo[15]);
         printf ("%lf %lf %lf %lf\n", foo[16], foo[17], foo[18], foo[19]);
      }
   }
}

void gen_phs_from_x_gpu (double sqrts, phs_dim_t d, int n_channels, int *channel_lims,
                         int n_x, double *x_h, double *factors_h, double *volumes_h, int *oks_h, double *p_h) {

   cudaDeviceProp prop;
   cudaGetDeviceProperties (&prop, 0);
   gen_phs_from_x_gpu_batch (sqrts, d, n_channels, channel_lims, n_x, x_h, factors_h, volumes_h, oks_h, p_h); 

   
}

