#include <cmath>
#include <cstring>
#include <algorithm>

#include "phs.h"
#include "mappings.h"

#define MAP_INV_MASS 10

mapping_t *mappings_host = NULL;

static double *m_max = NULL;

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

void set_msq_cpu (phs_dim_t d, int channel, int branch_idx,
                  xcounter_t *xc, double sqrts, double *msq,
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
      if (id == 0) {
         printf ("x: %lf\n", x);
         printf ("map_id: %d\n", mappings_host[channel].map_id[branch_idx]);
      }      
      double f;
      double *a = mappings_host[channel].a[branch_idx].a;
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

static int first = 1;

void set_angles_cpu (phs_dim_t d, int channel, int branch_idx,
                     xcounter_t *xc, double s, double *msq, double *factor,
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
      if (first) printf ("xphi: %lf\n", x);
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
      if (first) {
         printf ("%lf %lf %lf %lf\n", L_new[0][0], L_new[0][1], L_new[0][2], L_new[0][3]);         
         printf ("%lf %lf %lf %lf\n", L_new[1][0], L_new[1][1], L_new[1][2], L_new[1][3]);         
         printf ("%lf %lf %lf %lf\n", L_new[2][0], L_new[2][1], L_new[2][2], L_new[2][3]);         
         printf ("%lf %lf %lf %lf\n", L_new[3][0], L_new[3][1], L_new[3][2], L_new[3][3]);         
      }
      
      set_angles_cpu (d, channel, k1, xc, s, msq, factor, p_decay, prt, L_new);
      set_angles_cpu (d, channel, k2, xc, s, msq, factor, p_decay, prt, L_new);
   }
}


void init_mapping_constants_cpu (int n_channels, double s, double msq_min, double msq_max) {
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

void gen_phs_from_x_cpu (double sqrts, phs_dim_t d, int n_x, double *x,
                         int *channels, double *factors, double *volumes, phs_prt_t *prt) {
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
      if (i == 0) {
         printf ("MSQ CPU: ");
         for (int j = 0; j < N_PRT; j++) {
            printf ("%lf ", msq[j]);
         }
         printf ("\n");

         printf ("DECAY CPU: ");
         for (int j = 0; j < N_PRT; j++) {
            printf ("%lf ", p_decay[j]);
         }
         printf ("\n");

      } 
      if (ok) set_angles_cpu (d, c, ROOT_BRANCH, &xc, sqrts * sqrts, msq, factors + i, p_decay, prt + N_PRT * i, L0);
      if (first) {
         printf ("%lf %lf %lf %lf\n", prt[0].p[0], prt[0].p[1], prt[0].p[2], prt[0].p[3]);
         printf ("%lf %lf %lf %lf\n", prt[1].p[0], prt[1].p[1], prt[1].p[2], prt[1].p[3]);
         printf ("%lf %lf %lf %lf\n", prt[2].p[0], prt[2].p[1], prt[2].p[2], prt[2].p[3]);
         printf ("%lf %lf %lf %lf\n", prt[3].p[0], prt[3].p[1], prt[3].p[2], prt[3].p[3]);
      }
      first = 0;
   }

   free (p_decay);
   free (msq);
   free (m_max);
}

