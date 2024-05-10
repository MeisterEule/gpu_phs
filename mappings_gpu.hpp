#ifndef MAPPINGS_GPU_HPP
#define MAPPINGS_GPU_HPP

#include "mappings.h"

__device__ mapping_t *mappings_d = NULL;

__device__ void mapping_msq_from_x_none (double x, double s, double m, double w, double msq_min, double msq_max, double *a,
                                         double *msq, double *factor) {
   *msq = (1 - x) * msq_min + x * msq_max;
   *factor = a[2];
}

__device__ void mapping_x_from_msq_none (double msq, double msq_min, double msq_max, double s, double m, double w,
                                         double *a, double *x, double *factor) {
  *x = (msq - msq_min) / a[2];
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

__device__ void mapping_x_from_msq_schannel (double msq, double msq_min, double msq_max, double s, double m, double w,
                                         double *a, double *x, double *factor) {
   double tmp = (msq - m * m) / (m * w);
   *x = (atan(tmp) - a[0]) / (a[1] - a[0]); 
   *factor = a[2] * (1 + tmp * tmp);
}

__device__ void mapping_msq_from_x_collinear (double x, double s, double m, double w, double msq_min, double msq_max, double *a,
                                              double *msq, double *factor) {
   double msq1 = a[0] * exp (x * a[1]);
   *msq = msq1 - a[0] + msq_min;
   *factor = a[2] * msq1;
}

__device__ void mapping_x_from_msq_collinear (double msq, double msq_min, double msq_max, double s, double m, double w,
                                              double *a, double *x, double *factor) {
  double tmp = msq - msq_min + a[0];
  *x = log(tmp / a[0]) / a[1];
  *factor = a[2] * tmp; 
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

__device__ void mapping_x_from_msq_tuchannel (double msq, double msq_min, double msq_max, double s, double m, double w,
                                              double *a, double *x, double *factor) {
  double tmp;
  if (msq < (msq_max + msq_min) / 2) {
     tmp = msq - msq_min + a[0];
     *x = log(tmp / a[0]) / a[1];
  } else {
     tmp = msq_max - msq + a[0];
     *x = 1 - log(tmp / a[0]) / a[1];
  } 
  *factor = a[2] * tmp;
}

__device__ void mapping_msq_from_x_step_e (double x, double s, double m, double w, double msq_min, double msq_max, double *a,
                                           double *msq, double *factor) {
   double tmp = exp (-x * a[0] / a[2]) * (1 + a[1]);
   double z = -a[2] * log(tmp - a[1]);
   *msq = z * msq_max + (1 - z) * msq_min;
   *factor = a[0] / (1 - a[1] / tmp) * (msq_max - msq_min) / s;
}

__device__ void mapping_x_from_msq_step_e (double msq, double msq_min, double msq_max, double s, double m, double w,
                                           double *a, double *x, double *factor) {
   double z = (msq - msq_min) / (msq_max - msq_min);
   double tmp = 1 + a[1] * exp (z / a[2]); 
   *x = (z - a[2] * log (tmp / (1 + a[1]))) / a[0];
   *factor = a[0] * tmp * (msq_max - msq_min) / s;
}

__device__ void mapping_msq_from_x_step_h (double x, double s, double m, double w, double msq_min, double msq_max, double *a,
                                           double *msq, double *factor) {
  double z = a[1] / (a[0] - x) - a[1] / a[0] + a[2] * x;
  *msq = z * msq_max + (1 - z) * msq_min;
  *factor = (a[1] / ((a[0] - x) * (a[0] - x)) + a[2]) * (msq_max - msq_min) / s;
}

__device__ void mapping_x_from_msq_step_h (double msq, double msq_min, double msq_max, double s, double m, double w,
                                           double *a, double *x, double *factor) {
   double z = (msq - msq_min) / (msq_max - msq_min);
   double tmp1 = a[1] / (a[0] * a[2]);
   double tmp2 = a[0] - z / a[2];
   *x = ((a[0] + z / a[2] + tmp1) - sqrt(tmp2 * tmp2 + 2 * tmp1 * (a[0] + z / a[2]) + tmp1 * tmp1)) / 2;
   *factor = (a[1] / (a[0] - *x) / (a[0] - *x) + a[2]) * (msq_max - msq_min) / s;
}

__device__ void mapping_ct_from_x_schannel (double x, double s, double *b, double *ct, double *st, double *factor) {
   double tmp = 2 * (1 - x);
   *ct = 1 - tmp;
   *st = sqrt (tmp * (2 - tmp));
   *factor = 1;
}

__device__ void mapping_x_from_ct_schannel (double ct, double st, double s, double *b, double *x, double *factor) {
   *x = (ct + 1) / 2;
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

__device__ void mapping_x_from_ct_collinear (double ct, double st, double s, double *b, double *x, double *factor) {
  double ct1;
  if (ct < 0) {
     ct1 = ct + b[0] + 1; 
     *x = log(ct1 / b[0]) / (2 * b[1]);
  } else {
     ct1 = -ct + b[0] + 1;
     *x = 1 - log(ct1 / b[0]) / (2 * b[1]);
  }
  *factor = ct1 * b[1];
}

__global__ void _init_mapping_constants (int n_channels, int n_part, double sqrts) {
   for (int c = 0; c < n_channels; c++) {
      double m_tot = mappings_d[c].mass_sum[0];
      for (int i = 0; i < n_part; i++) {
         double *a1 = mappings_d[c].a[i].a;
         double *a2 = mappings_d[c].a[i].a + 1;
         double *a3 = mappings_d[c].a[i].a + 2;
         double *b1 = mappings_d[c].b[i].a;
         double *b2 = mappings_d[c].b[i].a + 1;
         double *b3 = mappings_d[c].b[i].a + 2;
         double m = mappings_d[c].masses[i];
         double w = mappings_d[c].widths[i];
         double m_min = mappings_d[c].mass_sum[i];
         double m_max = sqrts - m_tot + m_min;
         double msq0 = m * m;
         double msq_min = m_min * m_min;
         double msq_max = m_max * m_max;
         double s = sqrts * sqrts;
         if (mappings_d[c].map_id[i] == MAP_SCHANNEL || 
             mappings_d[c].map_id[i] == MAP_STEP_E ||
             mappings_d[c].map_id[i] == MAP_STEP_H) {
            if (msq0 < msq_min || msq0 > msq_max) {
               mappings_d[c].map_id[i] = MAP_NO;
            }
         }
         int map_id = mappings_d[c].map_id[i];
         // Compute a for msq
         switch (map_id) {
            case MAP_NO:
               *a1 = 0;
               *a2 = msq_max - msq_min;
               *a3 = *a2 / s;
               break;
            case MAP_SCHANNEL:
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
            mappings_d[c].comp_msq_inv[i] = mapping_x_from_msq_none;
            mappings_d[c].comp_ct[i] = mapping_ct_from_x_schannel;
            mappings_d[c].comp_ct_inv[i] = mapping_x_from_ct_schannel;
            break;
         case MAP_SCHANNEL:
            mappings_d[c].comp_msq[i] = mapping_msq_from_x_schannel;
            mappings_d[c].comp_msq_inv[i] = mapping_x_from_msq_schannel;
            mappings_d[c].comp_ct[i] = mapping_ct_from_x_schannel;
            mappings_d[c].comp_ct_inv[i] = mapping_x_from_ct_schannel;
            break;
         case MAP_COLLINEAR:
            mappings_d[c].comp_msq[i] = mapping_msq_from_x_collinear;
            mappings_d[c].comp_msq_inv[i] = mapping_x_from_msq_collinear;
            mappings_d[c].comp_ct[i] = mapping_ct_from_x_collinear;
            mappings_d[c].comp_ct_inv[i] = mapping_x_from_ct_collinear;
            break;
         case MAP_RADIATION:
         case MAP_INFRARED:
            mappings_d[c].comp_msq[i] = mapping_msq_from_x_collinear;
            mappings_d[c].comp_msq_inv[i] = mapping_x_from_msq_collinear;
            mappings_d[c].comp_ct[i] = mapping_ct_from_x_schannel;
            mappings_d[c].comp_ct_inv[i] = mapping_x_from_ct_schannel;
            break;
         case MAP_UCHANNEL:
         case MAP_TCHANNEL:
            mappings_d[c].comp_msq[i] = mapping_msq_from_x_tuchannel;
            mappings_d[c].comp_msq_inv[i] = mapping_x_from_msq_tuchannel;
            mappings_d[c].comp_ct[i] = mapping_ct_from_x_collinear;
            mappings_d[c].comp_ct_inv[i] = mapping_x_from_ct_collinear;
            break;
         case MAP_STEP_E:
            mappings_d[c].comp_msq[i] = mapping_msq_from_x_step_e;
            mappings_d[c].comp_msq_inv[i] = mapping_x_from_msq_step_e;
            mappings_d[c].comp_ct[i] = mapping_ct_from_x_schannel;
            mappings_d[c].comp_ct_inv[i] = mapping_x_from_ct_schannel;
            break;
         case MAP_STEP_H:
            mappings_d[c].comp_msq[i] = mapping_msq_from_x_step_h;
            mappings_d[c].comp_msq_inv[i] = mapping_x_from_msq_step_h;
            mappings_d[c].comp_ct[i] = mapping_ct_from_x_schannel;
            mappings_d[c].comp_ct_inv[i] = mapping_x_from_ct_schannel;
   }
}


#endif
