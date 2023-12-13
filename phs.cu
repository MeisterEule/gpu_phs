#include "phs.h"
#include "monitoring.h"

int N_PRT = 0;
int N_PRT_IN = 0;
int N_PRT_OUT = 0;
int PRT_STRIDE = 0;
int ROOT_BRANCH = 0;
int N_PART = 0;

__device__ int DN_PRT;
__device__ int DN_PRT_OUT;
__device__ int DN_PART;
__device__ int DPRT_STRIDE;
__device__ int DROOT_BRANCH;

#include "mappings_gpu.hpp"

int **daughters1 = NULL;
int **daughters2 = NULL;
int **has_children = NULL;

static double *m_max = NULL;

__global__ void _init_mappings (int n_channels, mapping_t *map_h) {
   mappings_d = (mapping_t*)malloc(n_channels * sizeof(mapping_t));
   for (int c = 0; c < n_channels; c++) {
      mappings_d[c].map_id = (int*)malloc(DN_PRT_OUT * sizeof(int));
      mappings_d[c].comp_ct = (mapping_ct_sig**)malloc(DN_PRT_OUT * sizeof(mapping_ct_sig*));
      mappings_d[c].comp_msq = (mapping_msq_sig**)malloc(DN_PRT_OUT * sizeof(mapping_msq_sig*));
      mappings_d[c].a = (map_constant_t*)malloc(DN_PRT_OUT * sizeof(map_constant_t));
      mappings_d[c].b = (map_constant_t*)malloc(DN_PRT_OUT * sizeof(map_constant_t));
      mappings_d[c].masses = (double *)malloc(DN_PRT_OUT * sizeof(double));
      mappings_d[c].widths = (double *)malloc(DN_PRT_OUT * sizeof(double));
      for (int i = 0; i < DN_PRT_OUT; i++) {
         mappings_d[c].comp_ct[i] = NULL;
         mappings_d[c].comp_msq[i] = NULL;
      }
   }
}

__global__ void _fill_mapids (int channel, int *map_ids) {
   for (int i = 0; i < DN_PRT_OUT; i++) {
      mappings_d[channel].map_id[i] = map_ids[i];
   }
}

__global__ void _fill_masses (int channel, double *m, double *w) {
   for (int i = 0; i < DN_PRT_OUT; i++) {
      mappings_d[channel].masses[i] = m[i];
      mappings_d[channel].widths[i] = w[i];
   }
}

void set_mappings (int channel) {
   for (int i = 0; i < N_PRT_OUT; i++) {
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
   factors[DN_PRT * tid + DROOT_BRANCH] *= f;

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

__global__ void _init_phs (int _n_prt, int _n_prt_out, int _prt_stride, int _root_branch) {
  DN_PRT = _n_prt;
  DN_PRT_OUT = _n_prt_out;
  DPRT_STRIDE = _prt_stride;
  DROOT_BRANCH = _root_branch;
}

void set_angles_dummy (int *nboost_max, int *nboost, int branch_idx) {
   (*nboost)++;
   if (*nboost > *nboost_max) *nboost_max = *nboost;
   if (has_children[0][branch_idx]) {
      int k1 = daughters1[0][branch_idx];
      int k2 = daughters2[0][branch_idx];
      (*nboost)++;
      if (*nboost > *nboost_max) *nboost_max = *nboost;
      set_angles_dummy (nboost_max, nboost, k1);
      set_angles_dummy (nboost_max, nboost, k2);
      (*nboost)--;
   }
   (*nboost)--;
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
   set_angles_dummy (&n_boost_max, &dummy, ROOT_BRANCH);
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

void init_phs_gpu (int n_channels, mapping_t *map_h, double s) {
   _init_phs<<<1,1>>>(N_PRT, N_PRT_OUT, PRT_STRIDE, ROOT_BRANCH);

   cudaDeviceSynchronize();
   _init_mappings<<<1,1>>>(n_channels, map_h);
   int *tmp;
   cudaMalloc((void**)&tmp, N_PRT_OUT * sizeof(int));
   for (int c = 0; c < n_channels; c++) {
       cudaMemcpy (tmp, map_h[c].map_id, N_PRT_OUT * sizeof(int), cudaMemcpyHostToDevice);
       _fill_mapids<<<1,1>>> (c, tmp);
   }
   cudaFree(tmp);
   double *m, *w;
   cudaMalloc((void**)&m, N_PRT_OUT * sizeof(double));
   cudaMalloc((void**)&w, N_PRT_OUT * sizeof(double));
   for (int c = 0; c < n_channels; c++) {
       cudaMemcpy (m, map_h[c].masses, N_PRT_OUT * sizeof(double), cudaMemcpyHostToDevice);
       cudaMemcpy (w, map_h[c].widths, N_PRT_OUT * sizeof(double), cudaMemcpyHostToDevice);
       _fill_masses<<<1,1>>> (c, m, w);
   }
   cudaFree(m);
   cudaFree(w);
   cudaDeviceSynchronize();
   _init_mapping_constants<<<1,1>>> (n_channels, s, 0, s);
   for (int c = 0; c < n_channels; c++) {
      set_mappings(c);
   }
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
     double *a = mappings_d[channel].a[branch_idx].a;
     double m = mappings_d[channel].masses[branch_idx];
     double w = mappings_d[channel].widths[branch_idx];
     double f;
     ///mappings_d[channel].comp_msq[branch_idx](x, sqrts * sqrts, m, w, msq_min, msq_max, a,
     ///                                         &msq[DN_PART * tid + branch_idx], &f);
     ///factors[DN_PART * tid + branch_idx] = f * factors[DN_PART * tid + k1] * factors[DN_PART * tid + k2]; 
     ///volumes[DN_PART * tid + branch_idx] *= volumes[DN_PART * tid + k1] * volumes[DN_PART * tid + k2] * sqrts * sqrts / (4 * TWOPI2);
  }
  // ROOT BRANCH
  int k1 = cmd[3 * n_cmd * channel + 3 * (n_cmd-1)];
  int k2 = cmd[3 * n_cmd * channel + 3 * (n_cmd-1) + 1];
  //msq[DN_PART * tid + DROOT_BRANCH] = sqrts * sqrts;
  //factors[DN_PART * tid + DROOT_BRANCH] = factors[DN_PART * tid + k1] * volumes[DN_PART * tid + k2] / (4 * TWOPI5);
}

__global__ void _apply_decay (int N, double sqrts, int channel, int *cmd, int n_cmd,
                              double *msq, double *p_decay, double *factors) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= N) return;
  for (int c = 0; c < n_cmd; c++) {
     int k1 = cmd[3 * n_cmd * channel + 3 * c];
     int k2 = cmd[3 * n_cmd * channel + 3 * c + 1];
     int branch_idx = cmd[3 * n_cmd * channel + 3 * c + 2]; 
     double msq0 = msq[DN_PART * tid + branch_idx];
     double msq1 = msq[DN_PART * tid + k1];
     double msq2 = msq[DN_PART * tid + k2];
     double m0 = sqrt(msq0);
     double m1 = sqrt(msq1);
     double m2 = sqrt(msq2);
     double lda = (msq0 - msq1 - msq2) * (msq0 - msq1 - msq2) - 4 * msq1 * msq2; 
     p_decay[DN_PART * tid + k1] = sqrt(lda) / (2 * m0);
     p_decay[DN_PART * tid + k2] = -sqrt(lda) / (2 * m0);
     factors[DN_PART * tid + branch_idx] *= sqrt(lda) / msq0;
  }
}

void gen_phs_from_x_gpu_2 (phs_dim_t d, int *cmds, int n_cmds, 
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

   int *cmds_d;
   cudaMalloc((void**)&cmds_d, 3 * n_channels * n_cmds * sizeof(int));
   cudaMemcpy(cmds_d, cmds, 3 * n_channels * n_cmds * sizeof(int), cudaMemcpyHostToDevice);

   double *msq_d;
   cudaMalloc((void**)&msq_d, N_PART * d.n_events_gen * sizeof(double)); 
   double *p_decay;
   cudaMalloc((void**)&p_decay, N_PART * d.n_events_gen * sizeof(double));

   double *factors_d;
   cudaMalloc ((void**)&factors_d, N_PRT * d.n_events_gen * sizeof(double));
   double *volumes_d;
   cudaMalloc ((void**)&volumes_d, N_PRT * d.n_events_gen * sizeof(double));
   int *oks_d;
   cudaMalloc ((void**)&oks_d, N_PRT * d.n_events_gen * sizeof(int));
   _init_fv<<<d.n_events_gen/1024 + 1,1024>>> (d.n_events_gen, factors_d, volumes_d, oks_d);

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
      _apply_msq<<<nb,nt>>>(d.batch[channel], sqrts, channel, cmds_d, n_cmds, xc, msq_d, factors_d, volumes_d);
      cudaDeviceSynchronize();
      //_apply_decay<<<nb,nt>>>(d.batch[channel], sqrts, channel, cmds_d, n_cmds, msq_d, p_decay, factors_d); 
      cudaDeviceSynchronize();
      printf ("Channel: %s\n", cudaGetErrorString(cudaGetLastError()));
   }
}

void gen_phs_from_x_gpu (double sqrts, phs_dim_t d, int n_channels, int *channel_lims,
                         int n_x, double *x_h, double *factors_h, double *volumes_h, int *oks_h, double *p_h) {

   cudaDeviceProp prop;
   cudaGetDeviceProperties (&prop, 0);
   gen_phs_from_x_gpu_batch (sqrts, d, n_channels, channel_lims, n_x, x_h, factors_h, volumes_h, oks_h, p_h); 

   
}

