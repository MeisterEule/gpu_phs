#ifndef MAPPINGS_GPU_HPP
#define MAPPINGS_GPU_HPP

#include "mappings.h"

#define PI 3.14159265358979323846 
#define TWOPI      6.28318530717958647693
#define TWOPI2    39.47841760435743447534
#define TWOPI5  9792.62991312900650440772

__device__ mapping_t *mappings_d = NULL;

__device__ void mapping_msq_from_x_none (double x, double s, double m, double w, double msq_min, double msq_max, double *a,
                                         double *msq, double *factor) {
   *msq = (1 - x) * msq_min + x * msq_max;
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


#endif
