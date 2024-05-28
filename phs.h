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
   int nx;
   size_t *id_gpu;
   int id_cpu;
   double *x;
} xcounter_t;

typedef struct {
   double p[4];
   double p2;
   bool defined;
} phs_prt_t;

typedef struct {
   phs_prt_t *prt;
   double *factors;
   double v;
   int ok;
} phs_val_t;

extern int N_EXT_IN;
extern int N_EXT_OUT;
extern int N_EXT_TOT;

extern int N_PRT;
extern int N_PART;
extern int N_PRT_OUT;
extern int PRT_STRIDE;
extern int ROOT_BRANCH;
extern int N_BRANCHES;
extern int N_MSQ;
extern int N_BOOSTS;
extern int N_BRANCHES_INTERNAL;
extern int N_LAMBDA_IN;
extern int N_LAMBDA_OUT; 

extern int **daughters1;
extern int **daughters2;
extern int **has_children;
extern int *contains_friends;

extern int **i_scatter;
extern int **i_gather;

extern mapping_t *mappings_host;
extern double *flv_masses;
extern double *flv_widths;

int search_in_igather (int c, int x);

void init_mapping_constants_cpu (int n_channels, double sqrts);
void set_mappings (int channel);
void init_phs_gpu (int n_channels, mapping_t *map_h, double s);

void gen_phs_from_x_gpu (size_t n_events, 
                         int n_channels, int *channel_lims, int n_x, double *x_h,
                         double *factors_h, double *volumes_h, bool *oks_h, double *p_h);

void gen_phs_from_x_cpu_time_and_check (double sqrts, size_t n_events, int n_x, double *x, int n_channels, int *channels,
                                        size_t *n_ok, double *p_gpu, double *factors_gpu, bool *oks_gpu, FILE *fp);



#endif
