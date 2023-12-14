#ifndef PHS_H
#define PHS_H

#include <stdio.h>

#include "mappings.h"

enum {MAP_NO=0,
      MAP_SCHANNEL=1,
      MAP_TCHANNEL=2,
      MAP_UCHANNEL=3,
      MAP_RADIATION=4,
      MAP_COLLINEAR=5,
      MAP_INFRARED=6,
      MAP_STEP_E=11,
      MAP_STEP_H=12,
      MAP_ONSHELL=99};

typedef struct {
   int n_events_val;
   int n_events_gen;
   int *nt; // n_threads
   int *nb; // n_blocks 
   int *batch;
} phs_dim_t;

typedef struct {
   int nx;
   int *id_gpu;
   int id_cpu;
   double *x;
} xcounter_t;

typedef struct {
   double p[4];
} phs_prt_t;

typedef struct {
   phs_prt_t *prt;
   double f;
   double v;
   int ok;
} phs_val_t;

extern int N_PRT;
extern int N_PART;
extern int N_PRT_OUT;
extern int PRT_STRIDE;
extern int ROOT_BRANCH;

extern int **daughters1;
extern int **daughters2;
extern int **has_children;

extern int **i_scatter;
extern int **i_gather;

extern mapping_t *mappings_host;

int search_in_igather (int c, int x);
long long count_gpu_memory_requirements (phs_dim_t d, int n_x);

void init_mapping_constants_cpu (int n_channels, double s, double msq_min, double msq_max);
void set_mappings (int channel);
void init_phs_gpu (int n_channels, mapping_t *map_h, double s);

void gen_phs_from_x_gpu (double sqrts, phs_dim_t d, int n_channels,
                         int *channel_lims, int n_x, double *x,
                         double *factors, double *volumes, int *oks, double *p);

void gen_phs_from_x_gpu_2 (phs_dim_t d, 
                           int n_channels, int *channel_lims, int n_x, double *x_h);

void gen_phs_from_x_cpu (double sqrts, phs_dim_t d, int n_x, double *x, int *channels, double *factors, double *volumes, phs_prt_t *prt);


#endif
