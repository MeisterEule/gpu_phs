#include "phs.h"

#define PI 3.14159265358979323846 
#define TWOPI      6.28318530717958647693
#define TWOPI2    39.47841760435743447534
#define TWOPI5  9792.62991312900650440772

int N_PRT = 0;
int N_PRT_IN = 0;
int N_PRT_OUT = 0;
int PRT_STRIDE = 0;
int ROOT_BRANCH = 0;

__device__ int DN_PRT;
__device__ int DN_PRT_OUT;
__device__ int DPRT_STRIDE;
__device__ int DROOT_BRANCH;

int **daughters1 = NULL;
int **daughters2 = NULL;
int **has_children = NULL;

static double *m_max = NULL;

#define MAP_INV_MASS 10
#define MAP_INV_WIDTH 10

mapping_t *mappings_host = NULL;

__device__ mapping_t *mappings_d = NULL;

__device__ void mapping_msq_from_x_none (double x, double s, double m, double w, double msq_min, double msq_max, double *a,
                                         double *msq, double *factor) {
   *msq = (1 - x) * msq_min + x * msq_max;
   //*factor = (msq_max - msq_min) / s;
   *factor = a[2];
}

__device__ void mapping_msq_from_x_schannel (double x, double s, double m, double w, double msq_min, double msq_max, double *a,
                                             double *msq, double *factor) {
   double z = (1 - x) * a[0] + x * a[1];
   if (-PI/2 < z && z < PI/2) {
      double tmp = tan(z);
      *msq = m * (m + w * tmp);
      *factor = a[2] * (1 + tmp * tmp);
   } else {
      *msq = 0;
      *factor = 0;
   }
}

__device__ void mapping_msq_from_x_collinear (double x, double s, double m, double w, double msq_min, double msq_max, double *a,
                                              double *msq, double *factor) {
   double msq1 = a[0] * exp (x * a[1]);
   *msq = msq1 - a[0] + msq_min;
   *factor = a[2] * msq1;
}

__device__ void mapping_msq_from_x_tuchannel (double x, double s, double m, double w, double msq_min, double msq_max, double *a,
                                              double *msq, double *factor) {
  double msq1;
  if (x < 0.5) {
     msq1 = a[0] * exp (x * a[1]);
     *msq = msq1 - a[0] + msq_min;
  } else {
     msq1 = a[0] * exp((1-x) * a[1]);
     *msq = -(msq1 - a[0]) + msq_max;
  }
  *factor = a[2] * msq1;
}

__device__ void mapping_msq_from_x_step_e (double x, double s, double m, double w, double msq_min, double msq_max, double *a,
                                           double *msq, double *factor) {
   double tmp = exp (-x * a[0] / a[2]) * (1 + a[1]);
   double z = -a[2] * log(tmp - a[1]);
   *msq = z * msq_max + (1 - z) * msq_min;
   *factor = a[0] / (1 - a[1] / tmp) * (msq_max - msq_min) / s;
}

__device__ void mapping_msq_from_x_step_h (double x, double s, double m, double w, double msq_min, double msq_max, double *a,
                                           double *msq, double *factor) {
  double z = a[1] / (a[0] - x) - a[1] / a[0] + a[2] * x;
  *msq = z * msq_max + (1 - z) * msq_min;
  *factor = (a[1] / ((a[0] - x) * (a[0] - x)) + a[2]) * (msq_max - msq_min) / s;
}


__device__ void mapping_ct_from_x_schannel (double x, double s, double *b, double *ct, double *st, double *factor) {
   double tmp = 2 * (1 - x);
   *ct = 1 - tmp;
   *st = sqrt (tmp * (2 - tmp));
   *factor = 1;
}

__device__ void mapping_ct_from_x_collinear (double x, double s, double *b, double *ct, double *st, double *factor) {
   double ct1;
   if (x < 0.5) {
      ct1 = b[0] * exp (2 * x * b[1]);
      *ct = ct1 - b[0] - 1;
   } else {
      ct1 = b[0] * exp (2 * (1 - x) * b[1]);
      *ct = -(ct1 - b[0]) + 1;
   }
   if (*ct >= -1 && *ct <= 1) {
      *st = sqrt(1 - *ct * *ct);
      *factor = ct1 * b[1];
   } else {
      *ct = 1;
      *st = 0;
      *factor = 0;
   }
}

void mapping_msq_from_x_cpu (int type, double x, double s, double msq_min, double msq_max, double *a, double *msq, double *factor) {
   double msq1;
   double tmp, z;
   switch (type) {
      case MAP_NO:
      case MAP_SCHANNEL:
      case MAP_INFRARED:
      case MAP_RADIATION:
      case MAP_STEP_H:
         *msq = (1 - x) * msq_min + x * msq_max;
         *factor = (msq_max - msq_min) / s;
         break;
      case MAP_COLLINEAR:
         msq1 = a[0] * exp (x * a[1]);
         *msq = msq1 - a[0] + msq_min;
         *factor = a[2] * msq1;
         break;
      case MAP_STEP_E:
         tmp = exp (-x * a[0] / a[2]) * (1 + a[1]);
         z = -a[2] * log (tmp - a[1]);
         *msq = z * msq_max + (1 - z) * msq_min;
         *factor = a[0] / (1 - a[1] / tmp) * (msq_max - msq_min) / s;
         break;
   }
}

void mapping_ct_from_x_cpu (int type, double x, double s, double *ct, double *st, double *factor) {
   double b1, b2, b3;
   double tmp;
   switch (type) {
      case MAP_NO:
      case MAP_SCHANNEL:
      case MAP_INFRARED:
      case MAP_RADIATION:
      case MAP_STEP_E:
      case MAP_STEP_H:
         tmp = 2 * (1 - x);
         *ct = 1 - tmp;
         *st = sqrt (tmp * (2 - tmp));
         *factor = 1; 
         break;
      case MAP_COLLINEAR:
         b1 = (double)(MAP_INV_MASS * MAP_INV_MASS) / s;
         b2 = log((b1 + 1) / b1);
         b3 = 0;
         if (x < 0.5) {
            tmp = b1 * exp (2 * x * b2);
            *ct = tmp - b1 - 1;
         } else {
            tmp = b1 * exp (2 * (1 - x) * b2);
            *ct = -(tmp - b1) + 1;
         }
         if (*ct >= -1 && *ct <= 1) {
            *st = sqrt(1 - *ct * *ct);
            *factor = tmp * b2;
         } else {
            *ct = 1;
            *st = 0;
            *factor = 0;
         }
         break;
   }
}

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

__global__ void _init_mapping_constants (int n_channels, double s, double msq_min, double msq_max) {
   double msq0;
   for (int c = 0; c < n_channels; c++) {
      for (int i = 0; i < DN_PRT_OUT; i++) {
         int map_id = mappings_d[c].map_id[i];
         double *a1 = mappings_d[c].a[i].a;
         double *a2 = mappings_d[c].a[i].a + 1;
         double *a3 = mappings_d[c].a[i].a + 2;
         double *b1 = mappings_d[c].b[i].a;
         double *b2 = mappings_d[c].b[i].a + 1;
         double *b3 = mappings_d[c].b[i].a + 2;
         double m = mappings_d[c].masses[i];
         double w = mappings_d[c].widths[i];
         // Compute a for msq
         switch (map_id) {
            case MAP_NO:
               *a1 = 0;
               *a2 = msq_max - msq_min;
               *a3 = *a2 / s;
               break;
            case MAP_SCHANNEL:
               msq0 = m * m;
               *a1 = atan ((msq_min - msq0) / (m * w));
               *a2 = atan ((msq_max - msq0) / (m * w));
               *a3 = (*a2 - (*a1)) * (m * w) / s;
               break;
            case MAP_RADIATION:
            case MAP_COLLINEAR:
            case MAP_INFRARED:
               if (map_id == MAP_RADIATION) {
                  msq0 = w * w;
               } else {
                  msq0 = m * m;
               }
               *a1 = msq0;
               *a2 = log((msq_max - msq_min) / msq0 + 1);
               *a3 = *a2 / s;
               break;
            case MAP_TCHANNEL:
            case MAP_UCHANNEL:
               msq0 = m * m;
               *a1 = msq0;
               *a2 = 2 * log ((msq_max - msq_min) / (2 * msq0) + 1);
               *a3 = *a2 / s;
               break;
            case MAP_STEP_E:
               *a3 = 2 * m * w / (msq_max - msq_min);
               if (*a3 < 0.01) *a3 = 0.01;
               *a2 = exp ( -(m * m - msq_min) / (msq_max - msq_min) / (*a3));
               *a1 = 1 - (*a3) * log ((1 + (*a2) * exp (1 / (*a3))) / (1 + (*a2)));
               break;
            case MAP_STEP_H:
               *a3 = (m * m - msq_min) / (msq_max - msq_min);
               *a2 = pow(2 * m * w / (msq_max - msq_min),2) / (*a3);
               if (*a2 < 0.000001) *a2 = 0.000001;
               *a1 = (1 + sqrt(1 + 4 * (*a2) / (1 - (*a3)))) / 2;
               break;
         }

         /// Compute b for ct
         switch (map_id) {
            case MAP_TCHANNEL:
            case MAP_UCHANNEL:
            case MAP_COLLINEAR:
               *b1 = m * m / s;
               *b2 = log((*b1 + 1) / (*b1));
               *b3 = 0;
               break;
            default:
               *b1 = 0;
               *b2 = 0;
               *b3 = 0;
               break;
         }
      }
   } 
} 

__global__ void _set_mappings (int c, int i) {
   switch (mappings_d[c].map_id[i]) {
         case MAP_NO:
            mappings_d[c].comp_msq[i] = mapping_msq_from_x_none;
            mappings_d[c].comp_ct[i] = mapping_ct_from_x_schannel;
            break;
         case MAP_SCHANNEL:
            mappings_d[c].comp_msq[i] = mapping_msq_from_x_schannel;
            mappings_d[c].comp_ct[i] = mapping_ct_from_x_schannel;
            break;
         case MAP_COLLINEAR:
            mappings_d[c].comp_msq[i] = mapping_msq_from_x_collinear;
            mappings_d[c].comp_ct[i] = mapping_ct_from_x_collinear;
            break;
         case MAP_RADIATION:
         case MAP_INFRARED:
            mappings_d[c].comp_msq[i] = mapping_msq_from_x_collinear;
            mappings_d[c].comp_ct[i] = mapping_ct_from_x_schannel;
            break;
         case MAP_UCHANNEL:
         case MAP_TCHANNEL:
            break;
            mappings_d[c].comp_msq[i] = mapping_msq_from_x_tuchannel;
            mappings_d[c].comp_ct[i] = mapping_ct_from_x_collinear;
         case MAP_STEP_E:
            mappings_d[c].comp_msq[i] = mapping_msq_from_x_step_e;
            mappings_d[c].comp_ct[i] = mapping_ct_from_x_schannel;
            break;
         case MAP_STEP_H:
            mappings_d[c].comp_msq[i] = mapping_msq_from_x_step_h;
            mappings_d[c].comp_ct[i] = mapping_ct_from_x_schannel;
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
   //if (!oks[DN_PRT * tid + branch_idx]) return;
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
   //if (channel == 1) printf ("FOO\n");
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

   if (branch_idx == ROOT_BRANCH) {
      cudaMemset(m_max, 0, d.batch * N_PRT * sizeof(double));
      _set_msq_root<<<d.nb,d.nt>>>(d.batch, branch_idx, k1, k2, msq + N_PRT * off, sqrts, factors + N_PRT * off, volumes + N_PRT * off, m_max);
   } else {
      _set_msq_branch<<<d.nb,d.nt>>>(d.batch, channel, off, branch_idx, k1, k2, xc, sqrts, msq + N_PRT * off,
                                     factors + N_PRT * off, volumes + N_PRT * off, oks + N_PRT * off, m_max);
   }
   cudaDeviceSynchronize();
   // if ok
   _set_decay<<<d.nb,d.nt>>> (d.batch, branch_idx, k1, k2, oks + N_PRT * off, msq + N_PRT * off, m_max, p_decay + N_PRT * off, factors + N_PRT * off);
}

void set_msq_cpu (phs_dim_t d, int channel, int branch_idx, xcounter_t *xc, double sqrts, double *msq,
                  double *factor, double *volume, bool *ok, double *p_decay) {
   int k1 = daughters1[channel][branch_idx];
   int k2 = daughters2[channel][branch_idx];
   double f1, f2, v1, v2;
   if (has_children[channel][k1]) {
      set_msq_cpu(d, channel, k1, xc, sqrts, msq, &f1, &v1, ok, p_decay);
      if (!(*ok)) return;
   } else {
      f1 = 1; v1 = 1;
   }
   if (has_children[channel][k2]) {
      set_msq_cpu(d, channel, k2, xc, sqrts, msq, &f2, &v2, ok, p_decay);
      if (!(*ok)) return;
   } else {
      f2 = 1; v2 = 1;
   }

   //printf ("branch_idx: %d\n", branch_idx);
   if (branch_idx == ROOT_BRANCH) {
      memset (m_max, 0, N_PRT * sizeof(double));
      msq[branch_idx] = sqrts * sqrts;
      m_max[branch_idx] = sqrts;
      *factor = f1 * f2;
      *volume = v1 * v2 / (4 * TWOPI5);
   } else {
      m_max[branch_idx] = sqrts;
      double msq_min = 0;
      double msq_max = sqrts * sqrts;
      double this_msq = 0;
      int id = xc->id_cpu++;
      double x = xc->x[id];
      double f;
      double *a = mappings_host[channel].a[branch_idx].a;
      //printf ("Check a: %lf %lf %lf\n", a[0], a[1], a[2]);
      mapping_msq_from_x_cpu (mappings_host[channel].map_id[branch_idx], x, sqrts * sqrts, msq_min, msq_max, a, &msq[branch_idx], factor);
      if (this_msq >= 0) {
         *factor *= f1 * f2;
         *volume = v1 * v2  * sqrts * sqrts / (4 * TWOPI2);
      } else {
         *ok = false;
      }
   }

   if (*ok) {
      double this_msq = msq[branch_idx];
      double msq1 = msq[k1];
      double msq2 = msq[k2];
      double m1 = sqrt(msq1);
      double m2 = sqrt(msq2);
      double m = sqrt(this_msq);
      double lda = (this_msq - msq1 - msq2) * (this_msq - msq1 - msq2) - 4 * msq1 * msq2;
      if (lda > 0 && m > m1 + m2 && m <= m_max[branch_idx]) {
        p_decay[k1] = sqrt(lda) / (2 * m);
        p_decay[k2] = -p_decay[k1];
        *factor *= sqrt(lda) / this_msq;
      } else {
        *ok = false;
      }
   }
}


__global__ void _apply_boost (int N, int branch_idx, int *oks, double *p_decay, double *msq, double *prt, double *L) {
   int tid = threadIdx.x + blockDim.x * blockIdx.x;
   if (tid >= N) return;
   //if (!oks[DN_PRT * tid + branch_idx]) return;
   double p = p_decay[DN_PRT * tid + branch_idx];
   double E = sqrt(msq[DN_PRT * tid + branch_idx] + p * p);
   if (tid == 0) printf ("apply_boost: %d, p: %lf, E: %lf\n", branch_idx, p, E);
   for (int i = 0; i < 4; i++) {
      prt[DPRT_STRIDE * tid + 4 * branch_idx + i] = L[16 * tid + 4 * i] * E + L[16 * tid + 4 * i + 3] * p;
   }
   if (tid == 0) printf ("prt[0]: %lf\n", prt[4 * branch_idx]);
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
   //if (!oks[DN_PRT * tid + branch_idx]) return;
   double p = p_decay[DN_PRT * tid + branch_idx];
   double m = sqrt(msq[DN_PRT * tid + branch_idx]);
   double bg = m > 0 ? p / m : 0;
   double gamma = sqrt (1 + bg * bg); 
   
   int gid = off + tid;
   //if (channel == 2) {
   //   printf ("gid: %d, %d \n", gid, xc->id_gpu[gid]);
   //}
   //if (gid == 57077) printf ("id: %d\n", xc->id_gpu[gid]);
   int xtid = xc->nx * gid + xc->id_gpu[gid]++; 
   //int xtid = 0;
   //if (channel == 2) printf ("tid: %d, gid: %d, id: %d, xtid: %d\n", tid, gid, xc->id_gpu[gid], xtid);
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
   double *L0_copy;
   cudaMalloc((void**)&L0_copy, 16 * d.batch * sizeof(double));
   cudaMemcpy(L0_copy, L0, 16 * d.batch * sizeof(double), cudaMemcpyDeviceToDevice);
   _apply_boost<<<d.nb,d.nt>>>(d.batch, branch_idx, oks + N_PRT * off, p_decay + N_PRT * off,
                               msq + N_PRT * off, prt + PRT_STRIDE * off, L0_copy);
   cudaDeviceSynchronize();
   if (has_children[channel][branch_idx]) {
      int k1 = daughters1[channel][branch_idx];
      int k2 = daughters2[channel][branch_idx];
      double *L_new;
      cudaMalloc((void**)&L_new, 16 * d.batch * sizeof(double));
      cudaMemset (L_new, 0, 16 * d.batch * sizeof(double));
      _create_new_boost<<<d.nb,d.nt>>>(d.batch, channel, off, branch_idx, xc, oks + N_PRT * off, s, p_decay + N_PRT * off,
                                       msq + N_PRT * off, factors + N_PRT * off, L0_copy, L_new);
      cudaDeviceSynchronize();

      set_angles_gpu (d, channel, off, k1, xc, oks, s, msq, factors, p_decay, prt, L_new);
      set_angles_gpu (d, channel, off, k2, xc, oks, s, msq, factors, p_decay, prt, L_new);
      cudaFree (L_new);
   }
   cudaFree (L0_copy); 
}

void set_angles_cpu (phs_dim_t d, int channel, int branch_idx, xcounter_t *xc, double s, double *msq, double *factor,
                     double *p_decay, phs_prt_t *prt, double L0[4][4]) {
   double p = p_decay[branch_idx];
   double m  = sqrt(msq[branch_idx]);
   double E = sqrt(msq[branch_idx] + p * p);
   for (int i = 0; i < 4; i++) {
      prt[branch_idx].p[i] = L0[i][0] * E + L0[i][3] * p;
   }

   if (has_children[channel][branch_idx]) {
      int k1 = daughters1[channel][branch_idx];
      int k2 = daughters2[channel][branch_idx];
      double bg = m > 0 ? p / m : 0;
      double gamma = sqrt (1 + bg * bg);
      int id = xc->id_cpu++;
      double x = xc->x[id];
      double phi = x * TWOPI;
      double cp = cos(phi);
      double sp = sin(phi);

      id = xc->id_cpu++;
      x = xc->x[id];
      double ct, st, f;
      mapping_ct_from_x_cpu (mappings_host[channel].map_id[branch_idx], x, s, &ct, &st, &f);
      *factor *= f; 

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
 
      double L_new[4][4]; 
      memset (L_new, 0, 16 * sizeof(double));
      for (int i = 0; i < 4; i++) {
         for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
               L_new[i][j] += L0[i][k] * L1[k][j];
            }
         }
      }
      
      set_angles_cpu (d, channel, k1, xc, s, msq, factor, p_decay, prt, L_new);
      set_angles_cpu (d, channel, k2, xc, s, msq, factor, p_decay, prt, L_new);
   }
}

void print_decay (int N, double *p_decay) {
   printf ("decay1: %lf %lf %lf %lf %lf\n", p_decay[0], p_decay[1], p_decay[2], p_decay[3], p_decay[4]);
   printf ("decay2: %lf %lf %lf %lf %lf\n", p_decay[5], p_decay[6], p_decay[7], p_decay[8], p_decay[9]);
}

void print_msq (int N, double *prt) {
   printf ("msq1: %lf %lf %lf %lf %lf\n", prt[N_PRT * 0 + 0], prt[N_PRT * 0 + 1], 
                                          prt[N_PRT * 0 + 2], prt[N_PRT * 0 + 3],
                                          prt[N_PRT * 0 + 4]);
   printf ("msq2: %lf %lf %lf %lf %lf\n", prt[N_PRT * 1 + 0], prt[N_PRT * 1 + 1], 
                                          prt[N_PRT * 1 + 2], prt[N_PRT * 1 + 3],
                                          prt[N_PRT * 1 + 4]);
}

void print_prt (int N, double *prt) {
   for (int i = 0; i < N_PRT; i++) {
      printf ("prt1[%d]: %lf %lf %lf %lf\n", i, prt[PRT_STRIDE * 0 + 4 * i],
                                                prt[PRT_STRIDE * 0 + 4 * i + 1],
                                                prt[PRT_STRIDE * 0 + 4 * i + 2],
                                                prt[PRT_STRIDE * 0 + 4 * i + 3]);
   }
   for (int i = 0; i < N_PRT; i++) {
      printf ("prt2[%d]: %lf %lf %lf %lf\n", i, prt[PRT_STRIDE * 1 + 4 * i],
                                                prt[PRT_STRIDE * 1 + 4 * i + 1],
                                                prt[PRT_STRIDE * 1 + 4 * i + 2],
                                                prt[PRT_STRIDE * 1 + 4 * i + 3]);

   }
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

#ifdef _VERBOSE
   printf ("GPU Memory requirements [GiB]: \n");
   printf ("   m_max: %lf\n", (double)mem_m_max / BYTES_PER_GB);
   printf ("   x_counter: %lf\n", (double)mem_xc / BYTES_PER_GB);
   printf ("   momenta tree: %lf\n", (double)mem_prt / BYTES_PER_GB);
   printf ("   p_decay: %lf\n", (double)mem_pdecay / BYTES_PER_GB);
   printf ("   msq: %lf\n", (double)mem_msq / BYTES_PER_GB);
   printf ("   factors: %lf\n", (double)mem_factors / BYTES_PER_GB);
   printf ("   volumes: %lf\n", (double)mem_volumes / BYTES_PER_GB);
   printf ("   oks: %lf\n", (double)mem_oks / BYTES_PER_GB);
   printf ("   Boosts: %lf\n", (double)mem_boost / BYTES_PER_GB);
   printf ("    Total: %lf\n", (double)mem_tot / BYTES_PER_GB);
#endif
   return mem_tot;
}

///void gen_phs_from_x_gpu_batch (double sqrts, phs_dim_t d, int channel, int channel_offset, 
///                               int n_x, double *x_h, double *factors_h, double *volumes_h, int *oks_h, double *p_h) {
void gen_phs_from_x_gpu_batch (double sqrts, phs_dim_t d, int n_channels, int *channel_offsets, 
                               int n_x, double *x_h, double *factors_h, double *volumes_h, int *oks_h, double *p_h) {

   cudaMalloc((void**)&m_max, N_PRT * d.batch * sizeof(double));

   double *x_d;
   int *id_d;
   //cudaMalloc((void**)&x_d, n_x * d.batch * sizeof(double));
   cudaMalloc((void**)&x_d, n_x * d.n_events_gen * sizeof(double));
   //cudaMalloc((void**)&id_d, d.batch * sizeof(int));
   cudaMalloc((void**)&id_d, d.n_events_gen * sizeof(int));
   xcounter_t *xc;
   cudaMalloc((void**)&xc, sizeof(xcounter_t));
   //cudaMemcpy (x_d, x_h + n_x * channel_offset, n_x * d.batch * sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy (x_d, x_h, n_x * d.n_events_gen * sizeof(double), cudaMemcpyHostToDevice);
   //cudaMemset (id_d, 0, d.batch * sizeof(int));
   cudaMemset (id_d, 0, d.n_events_gen * sizeof(int));
   _init_x<<<1,1>>> (xc, x_d, id_d, n_x);

   //memset (p_h + channel_offset * PRT_STRIDE, 0, PRT_STRIDE * d.batch * sizeof(double));
   memset (p_h, 0, PRT_STRIDE * d.n_events_gen * sizeof(double));
   double *p_d;
   //cudaMalloc((void**)&p_d, PRT_STRIDE * d.batch * sizeof(double));
   cudaMalloc((void**)&p_d, PRT_STRIDE * d.n_events_gen * sizeof(double));
   //cudaMemset(p_d, 0, PRT_STRIDE * d.batch * sizeof(double));
   cudaMemset(p_d, 0, PRT_STRIDE * d.n_events_gen * sizeof(double));

   double *pdecay_d;
   //cudaMalloc((void**)&pdecay_d, N_PRT * d.batch * sizeof(double));
   cudaMalloc((void**)&pdecay_d, N_PRT * d.n_events_gen * sizeof(double));
   //cudaMemset (pdecay_d, 0, N_PRT * d.batch * sizeof(double));
   cudaMemset (pdecay_d, 0, N_PRT * d.n_events_gen * sizeof(double));

   double *msq_d;
   //cudaMalloc((void**)&msq_d, N_PRT * d.batch * sizeof(double));
   //cudaMemset (msq_d, 0, N_PRT * d.batch * sizeof(double));
   cudaMalloc((void**)&msq_d, N_PRT * d.n_events_gen * sizeof(double));
   cudaMemset (msq_d, 0, N_PRT * d.n_events_gen * sizeof(double));

   double *factors_d;
   //cudaMalloc ((void**)&factors_d, N_PRT * d.batch * sizeof(double));
   cudaMalloc ((void**)&factors_d, N_PRT * d.n_events_gen * sizeof(double));

   double *volumes_d;
   //cudaMalloc ((void**)&volumes_d, N_PRT * d.batch * sizeof(double));
   cudaMalloc ((void**)&volumes_d, N_PRT * d.n_events_gen * sizeof(double));

   int *oks_d;
   //cudaMalloc ((void**)&oks_d, N_PRT * d.batch * sizeof(int));
   cudaMalloc ((void**)&oks_d, N_PRT * d.n_events_gen * sizeof(int));

   printf ("batch: %d\n", d.batch);
   _init_fv<<<d.n_events_gen/1024 + 1,d.nt>>> (d.batch, factors_d, volumes_d, oks_d);
   cudaDeviceSynchronize();
   for (int channel = 0; channel < n_channels; channel++) {
      int off = channel_offsets[channel];
      //printf ("channel: %d, off: %d\n", channel, off);
      set_msq_gpu (d, channel, off, ROOT_BRANCH, xc, sqrts,
                   msq_d, factors_d, volumes_d, oks_d, pdecay_d);
      cudaDeviceSynchronize();
#ifdef _VERBOSE
      printf ("CUDA msq %d: %s\n", channel, cudaGetErrorString(cudaGetLastError()));
#endif
   }

   //double *msq_h = (double*)malloc(N_PRT * d.n_events_gen * sizeof(double));
   //cudaMemcpy (msq_h, msq_d, N_PRT * d.n_events_gen * sizeof(double), cudaMemcpyDeviceToHost);
   //printf ("Check msq: ");
   //for (int i = 0; i < N_PRT; i++) {
   //   printf ("%lf ", msq_h[i]);
   //}
   //printf ("\n");

   //cudaMemcpy (factors_h + channel_offset * N_PRT, factors_d, N_PRT * d.batch * sizeof(double), cudaMemcpyDeviceToHost);
   //cudaMemcpy (factors_h, factors_d, N_PRT * d.batch * sizeof(double), cudaMemcpyDeviceToHost);
   double *L0;
   cudaMalloc((void**)&L0, 16 * d.batch * sizeof(double));
   _init_boost<<<d.nb,d.nt>>>(d.batch, L0);
   cudaDeviceSynchronize();
   for (int channel = 0; channel < n_channels; channel++) {
      int off = channel_offsets[channel];
      set_angles_gpu (d, channel, off, N_PRT_OUT - 1, xc, oks_d, sqrts * sqrts,
                      msq_d, factors_d, pdecay_d, p_d, L0);
      cudaDeviceSynchronize();
#ifdef _VERBOSE
      printf ("CUDA angles %d: %s\n", channel, cudaGetErrorString(cudaGetLastError()));
#endif
   }


   //cudaMemcpy (p_h + channel_offset * PRT_STRIDE, p_d, PRT_STRIDE * d.batch * sizeof(double), cudaMemcpyDeviceToHost);
   cudaMemcpy (p_h, p_d, PRT_STRIDE * d.n_events_gen * sizeof(double), cudaMemcpyDeviceToHost);
   //cudaMemcpy (factors_h + channel_offset * N_PRT, factors_d, N_PRT * d.batch * sizeof(double), cudaMemcpyDeviceToHost);
   cudaMemcpy (factors_h, factors_d, N_PRT * d.n_events_gen * sizeof(double), cudaMemcpyDeviceToHost);
   //cudaMemcpy (volumes_h + channel_offset * N_PRT, volumes_d, N_PRT * d.batch * sizeof(double), cudaMemcpyDeviceToHost);
   cudaMemcpy (volumes_h, volumes_d, N_PRT * d.n_events_gen * sizeof(double), cudaMemcpyDeviceToHost);
   //cudaMemcpy (oks_h + channel_offset * N_PRT, oks_d, N_PRT * d.batch * sizeof(int), cudaMemcpyDeviceToHost);
   cudaMemcpy (oks_h, oks_d, N_PRT * d.n_events_gen * sizeof(int), cudaMemcpyDeviceToHost);
   printf ("Check: %lf\n", p_h[0]);

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
}

void init_mapping_constants (int n_channels, double s, double msq_min, double msq_max) {
   double msq0;
   for (int c = 0; c < n_channels; c++) {
      for (int i = 0; i < N_PRT_OUT; i++) {
         int map_id = mappings_host[c].map_id[i];
         double *a1 = mappings_host[c].a[i].a;
         double *a2 = mappings_host[c].a[i].a + 1;
         double *a3 = mappings_host[c].a[i].a + 2;
         double *b1 = mappings_host[c].b[i].a;
         double *b2 = mappings_host[c].b[i].a + 1;
         double *b3 = mappings_host[c].b[i].a + 2;
         double m = mappings_host[c].masses[i];
         double w = mappings_host[c].widths[i];
         switch (map_id) {
            case MAP_COLLINEAR:
               //msq0 = (double)(MAP_INV_MASS * MAP_INV_MASS);
               msq0 = m * m;
               *a1 = msq0;
               *a2 = log((msq_max - msq_min) / msq0 + 1);
               *a3 = *a2 / s;
               *b1 = m * m / s;
               *b2 = log((*b1 + 1) / (*b1));
               *b3 = 0;
               break;
            case MAP_STEP_E:
               *a3 = std::max (2 * m * w / (msq_max - msq_min), 0.01);
               *a2 = exp ( -(m * m - msq_min) / (msq_max - msq_min) / (*a3));
               *a1 = 1 - (*a3) * log ((1 + (*a2) * exp (1 / (*a3))) / (1 + (*a2)));
               break;
            case MAP_NO:
            case MAP_SCHANNEL:
            case MAP_INFRARED:
            case MAP_RADIATION:
            case MAP_STEP_H:
               *a1 = 0;
               *a2 = 0;
               *a3 = 0;
               *b1 = 0;
               *b2 = 0;
               *b3 = 0;
               break;
         }
      }
   } 
}

void init_phs_gpu (int n_channels, mapping_t *map_h, double s) {
   printf ("Init phs GPU!\n");
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

void gen_phs_from_x_gpu (double sqrts, phs_dim_t d, int n_channels, int *channel_lims,
                         int n_x, double *x_h, double *factors_h, double *volumes_h, int *oks_h, double *p_h) {

   cudaDeviceProp prop;
   cudaGetDeviceProperties (&prop, 0);
   printf ("Max blocks: %d\n", prop.maxGridSize[0]);
   //for (int c = 0; c < n_channels; c++) {
   //   d.batch = channel_lims[c+1] - channel_lims[c];
   //   if (d.batch > 1024) {
   //      d.nt = 1024;
   //      d.nb = d.batch / 1024 + 1; 
   //   } else {
   //      d.nt = d.batch;
   //      d.nb = 1;
   //   }

   //   //set_mappings (c);
   //   cudaDeviceSynchronize();
   //   gen_phs_from_x_gpu_batch (sqrts, d, c, channel_lims[c], n_x, x_h, factors_h, volumes_h, oks_h, p_h); 
   //}

   d.batch = channel_lims[1];
   if (d.batch > 1024) {
      d.nt = 1024;
      d.nb = d.batch / 1024 + 1; 
   } else {
      d.nt = d.batch;
      d.nb = 1;
   }
   printf ("nblocks: %d\n", d.nb);
   gen_phs_from_x_gpu_batch (sqrts, d, n_channels, channel_lims, n_x, x_h, factors_h, volumes_h, oks_h, p_h); 

   
}

void gen_phs_from_x_cpu (double sqrts, phs_dim_t d, int n_x, double *x, int *channels, double *factors, double *volumes, phs_prt_t *prt) {
   double *p_decay = (double*)malloc(N_PRT * sizeof(double));
   double *msq = (double*)malloc(N_PRT * sizeof(double));
   m_max = (double*)malloc(N_PRT * sizeof(double));

   memset (prt, 0, N_PRT * 4 * d.n_events_gen * sizeof(double));
   memset (p_decay, 0, N_PRT * sizeof(double));
   memset (msq, 0, N_PRT * sizeof(double));
   memset (m_max, 0, N_PRT * sizeof(double));

   xcounter_t xc;
   xc.nx = n_x;
   xc.id_gpu = NULL;
   xc.id_cpu = 0;
   xc.x = x;

   double L0[4][4];
   memset (L0, 0, 16 * sizeof(double));
   L0[0][0] = 1;
   L0[1][1] = 1;
   L0[2][2] = 1;
   L0[3][3] = 1;

   for (int i = 0; i < d.n_events_gen; i++) {
      bool ok = true;
      int c = channels[i];
      set_msq_cpu (d, c, ROOT_BRANCH, &xc, sqrts, msq, factors + i, volumes + i, &ok, p_decay);
      if (ok) set_angles_cpu (d, c, ROOT_BRANCH, &xc, sqrts * sqrts, msq, factors + i, p_decay, prt + N_PRT * i, L0);
   }

   free (p_decay);
   free (msq);
   free (m_max);
}

