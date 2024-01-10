#ifndef MAPPINGS_H
#define MAPPINGS_H

#define PI 3.14159265358979323846 
#define TWOPI      6.28318530717958647693
#define TWOPI2    39.47841760435743447534
#define TWOPI5  9792.62991312900650440772

typedef void mapping_msq_sig (double, double, double, double, double, double, double*, double*, double*);
typedef void mapping_ct_sig (double, double, double *, double*, double*, double*);


typedef struct {
   double a[3];
} map_constant_t;

typedef struct {
   int *map_id;
   double *masses;
   double *widths;
   map_constant_t *a;
   map_constant_t *b;
   mapping_msq_sig **comp_msq;
   mapping_ct_sig **comp_ct;
   double *mass_sum;
} mapping_t;



#endif
