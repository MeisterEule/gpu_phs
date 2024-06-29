#include <cmath>
#include <cstring>
#include <algorithm>

#include "phs.h"
#include "global_phs.h"
#include "mappings.h"
#include "file_input.h"

#define MAP_INV_MASS 10

void mapping_msq_from_x_cpu (int type, double x, double s, double m, double w, double msq_min, double msq_max, double *a, double *msq, double *factor) {
   double msq1;
   double tmp, z;
   switch (type) {
      case MAP_NO:
         *msq = (1 - x) * msq_min + x * msq_max;
         *factor = (msq_max - msq_min) / s;
         break;
      case MAP_SCHANNEL:
        z = (1 - x) * a[0] + x * a[1];
        if (-PI/2 < z && z < PI/2) {
           tmp = tan(z);
           *msq = m * (m + w * tmp);
           *factor = a[2] * (1 + tmp * tmp);
        } else {
           *msq = 0;
           *factor = 0;
        }
        break;
      case MAP_COLLINEAR:
      case MAP_INFRARED:
      case MAP_RADIATION:
         msq1 = a[0] * exp (x * a[1]);
         *msq = msq1 - a[0] + msq_min;
         *factor = a[2] * msq1;
         break;
      case MAP_TCHANNEL:
      case MAP_UCHANNEL:
         if (x < 0.5) {
            msq1 = a[0] * exp (x * a[1]);
            *msq = msq1 - a[0] + msq_min;
         } else {
            msq1 = a[0] * exp((1-x) * a[1]);
            *msq = -(msq1 - a[0]) + msq_max;
         }
         *factor = a[2] * msq1;
         break;
      case MAP_STEP_H:
         z = a[1] / (a[0] - x) - a[1] / a[0] + a[2] * x;
         *msq = z * msq_max + (1 - z) * msq_min;
         *factor = (a[1] / ((a[0] - x) * (a[0] - x)) + a[2]) * (msq_max - msq_min) / s;
         break;
      case MAP_STEP_E:
         tmp = exp (-x * a[0] / a[2]) * (1 + a[1]);
         z = -a[2] * log (tmp - a[1]);
         *msq = z * msq_max + (1 - z) * msq_min;
         *factor = a[0] / (1 - a[1] / tmp) * (msq_max - msq_min) / s;
         break;
   }
}

void mapping_x_from_msq_cpu (int type, double s, double m, double w, double msq,
                             double msq_min, double msq_max, double *a, double *x, double *factor) {
   double msq0;
   double tmp;
   double z;
   switch (type) {
      case MAP_NO:
         *x = (msq - msq_min) / a[1];
         *factor = a[2];
         break;
      case MAP_SCHANNEL:
         msq0 = m*m;
         tmp = (msq - msq0) / (m * w);
         *x = (atan(tmp) - a[0]) / (a[1] - a[0]);
         *factor = a[2] * (1 + tmp*tmp);
         break;
      case MAP_COLLINEAR:
      case MAP_INFRARED:
      case MAP_RADIATION:
         tmp = msq - msq_min + a[0];
         *x = log (tmp / a[0]) / a[1];
         *factor = a[2] * tmp;
         break;
      case MAP_TCHANNEL:
      case MAP_UCHANNEL:
         if (msq < (msq_max + msq_min) / 2) {
            msq0 = msq - msq_min + a[0];
            *x = log(msq0 / a[0]) / a[1];
         } else {
            msq0 = msq_max - msq + a[0];
            *x = log(msq0 / a[0]) / a[1];
         }
         *factor = a[2] * msq0;
         break;
      case MAP_STEP_H:
         z = (msq - msq_min) / (msq_max - msq_min);
         tmp = a[1] / (a[0] * a[2]);
         *x = ((a[0] + z / a[2] + tmp) - sqrt((a[0] - z / a[2]) * (a[0] - z / a[2]) + 2 * tmp * (a[0] + z / a[2]) + tmp * tmp)) / 2;
         *factor = (a[1] / (a[0] - *x) / (a[0] - *x) + a[2]) * (msq_max - msq_min) / s;
         break;
      case MAP_STEP_E:
         z = (msq - msq_min) / (msq_max - msq_min); 
         tmp = 1 + a[1] * exp(z / a[2]);
         *x = (z - a[2] * log(tmp / (1 + a[1]))) / a[0];
         *factor = a[0] * tmp * (msq_max - msq_min) / s;
         break;
   }
}

void mapping_ct_from_x_cpu (int type, double x, double s, double *b, double *ct, double *st, double *factor) {
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
      case MAP_TCHANNEL:
      case MAP_UCHANNEL:
         if (x < 0.5) {
            tmp = b[0] * exp (2 * x * b[1]);
            *ct = tmp - b[0] - 1;
         } else {
            tmp = b[0] * exp (2 * (1 - x) * b[1]);
            *ct = -(tmp - b[0]) + 1;
         }
         if (*ct >= -1 && *ct <= 1) {
            *st = sqrt(1 - *ct * *ct);
            *factor = tmp * b[1];
         } else {
            *ct = 1;
            *st = 0;
            *factor = 0;
         }
         break;
   }
}

void mapping_x_from_ct_cpu (int type, double s, double ct, double st, double *b, double *x, double *factor) {
   double tmp;
   switch (type) {
      case MAP_NO:
      case MAP_SCHANNEL:
      case MAP_INFRARED:
      case MAP_RADIATION:
      case MAP_STEP_E:
      case MAP_STEP_H:
         *x = (ct + 1) / 2;
         *factor = 1;
         break;
      case MAP_COLLINEAR:
      case MAP_TCHANNEL:
      case MAP_UCHANNEL:
         if (ct < 0) {
            tmp = ct + b[0] + 1;
            *x = log(tmp / b[0]) / (2 * b[1]); 
         } else {
            tmp = -ct + b[0] + 1;
            *x = 1 - log(tmp / b[0]) / (2 * b[1]);
         }
         *factor = tmp * b[1];
   }
}

void set_msq_cpu (int channel, int branch_idx, double m_tot,
                  double *x, size_t *id_x, double sqrts, phs_prt_t *prt,
                  double *factor, double *volume, bool *ok, double *p_decay) {
   int k1 = daughters1[channel][branch_idx];
   int k2 = daughters2[channel][branch_idx];
   double f1, f2, v1, v2;
   if (has_children[channel][k1]) {
      set_msq_cpu(channel, k1, m_tot, x, id_x, sqrts, prt, &f1, &v1, ok, p_decay);
      if (!(*ok)) return;
   } else {
      f1 = 1; v1 = 1;
   }
   if (has_children[channel][k2]) {
      set_msq_cpu(channel, k2, m_tot, x, id_x, sqrts, prt, &f2, &v2, ok, p_decay);
      if (!(*ok)) return;
   } else {
      f2 = 1; v2 = 1;
   }

   double m_max;
   if (branch_idx == ROOT_BRANCH) {
      prt[branch_idx].p2 = sqrts * sqrts;
      m_max = sqrts;
      *factor = f1 * f2;
      *volume = v1 * v2 / (4 * TWOPI5);
   } else {
      double m_min = mappings_host[channel].mass_sum[branch_idx];
      m_max = sqrts - m_tot + m_min;
      double this_msq = 0;
      double f;
      double *a = mappings_host[channel].a[branch_idx].a;
      double m = mappings_host[channel].masses[branch_idx]; 
      double w = mappings_host[channel].widths[branch_idx];
      mapping_msq_from_x_cpu (mappings_host[channel].map_id[branch_idx], x[*id_x], sqrts * sqrts, m, w,
                              m_min*m_min, m_max*m_max, a, &prt[branch_idx].p2, factor);
      (*id_x)++;
      if (this_msq >= 0) {
         *factor *= f1 * f2;
         *volume = v1 * v2  * sqrts * sqrts / (4 * TWOPI2);
      } else {
         *ok = false;
      }
   }

   if (*ok) {
      double this_msq = prt[branch_idx].p2;
      double msq1 = prt[k1].p2;
      double msq2 = prt[k2].p2;
      double m1 = sqrt(msq1);
      double m2 = sqrt(msq2);
      double m = sqrt(this_msq);
      double lda = (this_msq - msq1 - msq2) * (this_msq - msq1 - msq2) - 4 * msq1 * msq2;
      if (lda > 0 && m > m1 + m2 && m <= m_max) {
        p_decay[k1] = sqrt(lda) / (2 * m);
        p_decay[k2] = -p_decay[k1];
        *factor *= sqrt(lda) / this_msq;
      } else {
        *ok = false;
      }
   }
}

void get_msq_cpu (int i_event, int channel, int branch_idx, double m_tot, phs_prt_t *prt, double sqrts,
                  double *p_decay, double *x, double *factor) {
   int k1 = daughters1[channel][branch_idx];
   int k2 = daughters2[channel][branch_idx];
   double f1, f2;
   if (has_children[channel][k1]) {
      get_msq_cpu (i_event, channel, k1, m_tot, prt, sqrts, p_decay, x, &f1);
   } else {
      f1 = 1;
   }
   if (has_children[channel][k2]) {
      get_msq_cpu (i_event, channel, k2, m_tot, prt, sqrts, p_decay, x, &f2);
   } else {
      f2 = 1;
   }

   double m_min = mappings_host[channel].mass_sum[branch_idx]; 
   double m_max = sqrts - m_tot + m_min;
   if (branch_idx == ROOT_BRANCH) {
      *factor = f1 * f2;
   } else {
      double *a = mappings_host[channel].a[branch_idx].a;
      double m = mappings_host[channel].masses[branch_idx]; 
      double w = mappings_host[channel].widths[branch_idx];
      mapping_x_from_msq_cpu (mappings_host[channel].map_id[branch_idx], sqrts * sqrts, m, w,
                              prt[branch_idx].p2, m_min*m_min, m_max*m_max, a, x, factor);
      *factor *= f1 * f2; 
   }

   double this_msq = prt[branch_idx].p2;
   double msq1 = prt[k1].p2;
   double msq2 = prt[k2].p2;
   double m1 = sqrt(msq1);
   double m2 = sqrt(msq2);
   double m = sqrt(this_msq);
   double lda = (this_msq - msq1 - msq2) * (this_msq - msq1 - msq2) - 4 * msq1 * msq2;
   if (lda > 0) {
      p_decay[k1] = sqrt(lda) / (2 * m);
      p_decay[k2] = -p_decay[k1];
      *factor *= sqrt(lda) / this_msq;
   } else {
      p_decay[k1] = 0;
      p_decay[k2] = 0;
      *factor = 0;
   }
}

void set_angles_cpu (int i_event, int channel, int branch_idx,
                     double *x, size_t *idx_x, double s, double *factor,
                     double *p_decay, phs_prt_t *prt, double L0[4][4]) {
   double p = p_decay[branch_idx];
   double m  = sqrt(prt[branch_idx].p2);
   double E = sqrt(prt[branch_idx].p2 + p * p);
   for (int i = 0; i < 4; i++) {
      prt[branch_idx].p[i] = L0[i][0] * E + L0[i][3] * p;
   }
   prt[branch_idx].defined = true;
   prt[branch_idx].p2 = prt[branch_idx].p[0] * prt[branch_idx].p[0] - prt[branch_idx].p[1] * prt[branch_idx].p[1]
                      - prt[branch_idx].p[2] * prt[branch_idx].p[2] - prt[branch_idx].p[3] * prt[branch_idx].p[3];

   if (has_children[channel][branch_idx]) {
      int k1 = daughters1[channel][branch_idx];
      int k2 = daughters2[channel][branch_idx];
      double bg = m > 0 ? p / m : 0;
      double gamma = sqrt (1 + bg * bg);
      double phi = x[*idx_x] * TWOPI;
      (*idx_x)++;
      double cp = cos(phi);
      double sp = sin(phi);

      double xx = x[*idx_x];
      (*idx_x)++;
      
      double *b = mappings_host[channel].b[branch_idx].a;
      double ct, st, f;
      mapping_ct_from_x_cpu (mappings_host[channel].map_id[branch_idx], xx, s, b, &ct, &st, &f);
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

      set_angles_cpu (i_event, channel, k1, x, idx_x, s, factor, p_decay, prt, L_new);
      set_angles_cpu (i_event, channel, k2, x, idx_x, s, factor, p_decay, prt, L_new);
   }
}

double azimuthal_angle_cpu (double n[3]) {
   return atan(n[1] / n[0]);
}

void polar_angle_ct_cpu (double n[3], double *ct, double *st) {
   double dn = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
   *ct = n[2] / dn;
   *st = sqrt(1 - (*ct)*(*ct));
}

void get_angles_cpu (int i_event, int channel, int branch_idx,
                double s, double *p_decay, phs_prt_t *prt,  
                double L0[4][4], double *phi0, double *ct0, double *st0, double *x, double *factor) {
   int k1 = daughters1[channel][branch_idx];
   int k2 = daughters2[channel][branch_idx];
   double p1[4];
   for (int i = 0; i < 4; i++) {
      p1[i] = 0;
      for (int j = 0; j < 4; j++) {
         p1[i] += L0[i][j] * prt[branch_idx].p[j];
      }
   }
   double cp0, sp0; 
   if (phi0 != NULL) {
      cp0 = cos(*phi0);  
      sp0 = sin(*phi0);  
   }
   double n[3];
   double ct = ct0 != NULL ? *ct0 : 0;
   double st = st0 != NULL ? *st0 : 0;
   double m = sqrt(prt[branch_idx].p2);
   double bg = m > 0 ? p_decay[branch_idx] / m : 0;
   double gamma = sqrt(1 + bg*bg);
   if (phi0 != NULL) {
      double px = cp0 * p1[1] + sp0 * p1[3];
      double py = -sp0 * p1[1] + cp0 * p1[2];
      n[0] = ct * px - st * p1[3];
      n[1] = py;
      n[2] = st * px + ct * p1[3];
      n[2] = n[2] * gamma - p1[0] * bg;
   } else {
      n[0] = p1[0];
      n[1] = p1[1];
      n[2] = p1[2];
      n[2] = n[2] * gamma - p1[0] * bg;
   }
   double phi = azimuthal_angle_cpu (n);
   polar_angle_ct_cpu (n, &ct, &st);
   double *b = mappings_host[channel].b[branch_idx].a;
   double f;
   mapping_x_from_ct_cpu (mappings_host[channel].map_id[branch_idx], s, ct, st, b, x, &f);
   *factor *= f;
 
   double L1[4][4];
   double L_new[4][4];
   memset (L_new, 0, 16 * sizeof(double));
   if (phi0 != NULL) {
      L1[0][0] = gamma;
      L1[0][1] = -bg * (*st0) * cp0;  
      L1[0][2] = -bg * (*st0) * sp0;
      L1[0][3] = -bg * (*ct0);
      L1[1][0] = 0;
      L1[1][1] = (*ct0) * cp0;
      L1[1][2] = (*ct0) * sp0;
      L1[1][3] = -(*st0);
      L1[2][0] = 0;
      L1[2][1] = -sp0;
      L1[2][2] = cp0;
      L1[2][3] = 0;
      L1[3][0] = -bg;
      L1[3][1] = gamma * (*st0) * cp0;
      L1[3][2] = gamma * (*st0) * sp0;
      L1[3][3] = gamma * (*ct0);

      for (int i = 0; i < 4; i++) {
         for (int j = 0; j < 4; j++) {
             for (int k = 0; k < 4; k++) {
                L_new[i][j] += L0[i][k] * L1[k][j];
             }
         }
      }
   } else {
      double gamma = sqrt(1 + bg*bg);
      L_new[0][0] = gamma; 
      L_new[0][3] = -bg;
      L_new[3][0] = -bg;
      L_new[3][3] = gamma;
   }
   if (has_children[channel][k1]) get_angles_cpu (i_event, channel, k1, s, p_decay, prt, L_new, &phi, &ct, &st, x, factor);
   if (has_children[channel][k2]) get_angles_cpu (i_event, channel, k2, s, p_decay, prt, L_new, &phi, &ct, &st, x, factor);
}

void combine_particles_cpu (int i_event, int channel, int branch_idx, phs_prt_t *prt) {
   int k1 = daughters1[channel][branch_idx];
   int k2 = daughters2[channel][branch_idx];
   if (has_children[channel][k1]) combine_particles_cpu (i_event, channel, k1, prt);
   if (has_children[channel][k2]) combine_particles_cpu (i_event, channel, k2, prt);
   for (int i = 0; i < 4; i++) {
      prt[branch_idx].p[i] = prt[k1].p[i] + prt[k2].p[i];
   }
   prt[branch_idx].p2 = prt[branch_idx].p[0] * prt[branch_idx].p[0] - prt[branch_idx].p[1] * prt[branch_idx].p[1]
                      - prt[branch_idx].p[2] * prt[branch_idx].p[2] - prt[branch_idx].p[3] * prt[branch_idx].p[3];
}


void init_mapping_constants_cpu (int n_channels, double sqrts) {
   double msq0;
   for (int c = 0; c < n_channels; c++) {
      double m_tot = mappings_host[c].mass_sum[ROOT_BRANCH];
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
         double m_min = mappings_host[c].mass_sum[i];
         double m_max = sqrts - m_tot + m_min;
         double msq_min = m_min * m_min;
         double msq_max = m_max * m_max;
         double s = sqrts * sqrts;
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
               *a3 = std::max (2 * m * w / (msq_max - msq_min), 0.01);
               *a2 = exp ( -(m * m - msq_min) / (msq_max - msq_min) / (*a3));
               *a1 = 1 - (*a3) * log ((1 + (*a2) * exp (1 / (*a3))) / (1 + (*a2)));
               break;
            case MAP_STEP_H:
               *a3 = (m * m - msq_min) / (msq_max - msq_min);
               *a2 = std::max (pow(2 * m * w / (msq_max - msq_min), 2) / (*a3), 0.000001);
               *a1 = (1 + sqrt(1 + 4 * (*a2) / (1 - (*a3)))) / 2; 
               break;
         }

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

void init_msq_cpu (double *msq) {
   int p = 1;
   for (int i = 0; i < N_EXT_OUT; i++) {
      msq[p-1] = flv_masses[N_EXT_IN+i] * flv_masses[N_EXT_IN+i];
      p *= 2;
   } 
}

void init_msq_cpu (phs_prt_t *prt) {
   int p = 1;
   for (int i = 0; i < N_EXT_OUT; i++) {
      prt[p-1].p2 = flv_masses[N_EXT_IN+i] * flv_masses[N_EXT_IN+i];
      p *= 2;
   }
}

#define BYTES_PER_GB 1073741824
void gen_phs_from_x_cpu_time_and_check (double sqrts, size_t n_events, int n_x, double *x,
                                        int n_channels, int *channels, size_t *n_oks,
                                        double *p_gpu, double *factors_gpu, bool *oks_gpu,
                                        FILE *fp) {
   double *p_decay = (double*)malloc(N_PRT * sizeof(double));
   double *msq = (double*)malloc(N_PRT * sizeof(double));

// The CPU routine should mimic the corresponding one in Whizard as closely as possible.
// For this reason, no contiguous 1D array is used, but the data structure "prt" from Whizard
// is reused. It has N_PRT elements.
   phs_prt_t *prt = (phs_prt_t*)malloc(N_PRT * sizeof(phs_prt_t));

   double factor, volume;
   bool ok;

   memset (p_decay, 0, N_PRT * sizeof(double));
   memset (msq, 0, N_PRT * sizeof(double));

   double L0[4][4];
   memset (L0, 0, 16 * sizeof(double));
   L0[0][0] = 1;
   L0[1][1] = 1;
   L0[2][2] = 1;
   L0[3][3] = 1;

   *n_oks = 0;
   double *all_factors = (double*)malloc(n_channels * sizeof(double));
   int n_events_failed = 0;
   double epsilon = input_control.compare_tolerance;
   double max_abs_deviation[4] = {0, 0, 0, 0};
   double sum_abs_deviation[4] = {0, 0, 0, 0};

   for (size_t i = 0; i < n_events; i++) {
      ok = true;
      int this_channel = channels[i];
      ///memset (msq, 0, N_PRT * sizeof(double));
      for (int j = 0; j < N_PRT; j++) {
         msq[j] = -1;
      }
      memset (p_decay, 0, N_PRT * sizeof(double));
      memset (prt, 0, 5 * N_PRT * sizeof(double));
      init_msq_cpu (prt);
      size_t id_x = 0;
      double m_tot = mappings_host[this_channel].mass_sum[ROOT_BRANCH];
      set_msq_cpu (this_channel, ROOT_BRANCH, m_tot, x + n_x * i, &id_x, sqrts, prt, &factor, &volume, &ok, p_decay); 
      if (ok) {
         set_angles_cpu (i, this_channel, ROOT_BRANCH, x + n_x * i, &id_x, sqrts * sqrts, &factor, p_decay, prt, L0);
      } else {
        //printf ("Not ok CPU: %d\n", i);
      }
      all_factors[this_channel] = factor;

      // The runtime check does not have a large impact on the measured time, but it is observable.
      // With this flag, we can switch it off to get the most reliable result.
      // That one if statement which remains does not make a difference.

      if (input_control.check_cpu && ok) {
         for (int n = 0; n < N_EXT_OUT; n++) {
            double *p = &p_gpu[4*N_EXT_OUT*i + 4*n];
            int nn = pow(2,n) - 1;
            if (fabs (p[0] - prt[nn].p[0]) > epsilon 
             || fabs (p[1] - prt[nn].p[1]) > epsilon  
             || fabs (p[2] - prt[nn].p[2]) > epsilon  
             || fabs (p[3] - prt[nn].p[3]) > epsilon) {
               fprintf (fp, "Error in p%d (event: %ld, channel: %d):\n", n, i, this_channel);
               fprintf (fp, "GPU: %lf %lf %lf %lf\n", p[0], p[1], p[2], p[3]);
               fprintf (fp, "CPU:  %lf %lf %lf %lf\n", prt[nn].p[0], prt[nn].p[1],
                                                       prt[nn].p[2], prt[nn].p[3]);
               for (int j = 0; j < 4; j++) {
                  double d = fabs(p[j] - prt[nn].p[j]); 
                  if (d > max_abs_deviation[j]) max_abs_deviation[j] = d;
                  sum_abs_deviation[j] += d;
               }
               n_events_failed++;
            }
        }
      }
      if (ok) (*n_oks)++;

      for (int j = 0; j < N_PRT; j++) {
         if (msq[j] < 0) {
            msq[j] = prt[j].p[0]*prt[j].p[0] - prt[j].p[1]*prt[j].p[1]
                   - prt[j].p[2]*prt[j].p[2] - prt[j].p[3]*prt[j].p[3];   
            ///if (i == 0) printf ("Set msq[%d]: %lf\n", j, msq[j]);
         }
      }
      if (input_control.do_inverse_mapping) {
         for (int c = 0; c < n_channels; c++) {
            if (c == this_channel) continue;
            combine_particles_cpu (i, c, ROOT_BRANCH, prt);
         }

         for (int c = 0; c < n_channels; c++) {
            if (c == this_channel) continue;
            m_tot = mappings_host[c].mass_sum[ROOT_BRANCH];
            double x, factor;
            get_msq_cpu (i, c, ROOT_BRANCH, m_tot, prt, sqrts, p_decay, &x, &factor);
            get_angles_cpu (i, c, ROOT_BRANCH, sqrts*sqrts, p_decay, prt, L0, NULL, NULL, NULL, &x, &factor);
            all_factors[c] = factor;
         }
      }
      ///if (i == 0) {
      ///   printf ("all_factors: ");
      ///   for (int c = 0; c < n_channels; c++) {
      ///      printf ("%lf ", all_factors[c]);
      ///   }
      ///   printf ("\n");
      ///}

      if (input_control.check_cpu && ok) {
         for (int c = 0; c < n_channels; c++) {
            if (fabs(all_factors[c] - factors_gpu[n_channels * i + c]) > epsilon) {
               fprintf (fp, "Error in factor (event: %d, channel: %d): \n", i, c);
               fprintf (fp, "GPU: %lf, CPU: %lf\n", factors_gpu[c], all_factors[c]);
            } 
         } 
      }
  }
  //printf ("Event generation done\n");
  free (all_factors);
  free (p_decay);
  free (msq);
  free (prt);

  fprintf (fp, "Failed events with EPSILON = %lf: %d / %d (%.2f%%)\n", epsilon, n_events_failed, n_events, (double)n_events_failed / n_events * 100);
  if (n_events_failed > 0) {
     fprintf (fp, "Max. deviations: %lf %lf %lf %lf\n",
              max_abs_deviation[0], max_abs_deviation[1], max_abs_deviation[2], max_abs_deviation[3]);
     fprintf (fp, "Avg. deviations: %lf %lf %lf %lf\n",
              sum_abs_deviation[0] / n_events_failed, sum_abs_deviation[1] / n_events_failed,
              sum_abs_deviation[2] / n_events_failed, sum_abs_deviation[3] / n_events_failed);
  }
}

