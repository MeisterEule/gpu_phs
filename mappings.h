#ifndef MAPPINGS_H
#define MAPPINGS_H

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
} mapping_t;



#endif
