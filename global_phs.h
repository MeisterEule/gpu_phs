#include "mappings.h"

extern int N_CHANNELS;
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

extern mapping_t *mappings_host;
extern double *flv_masses;
extern double *flv_widths;

typedef struct {
  int msq_threads;
  int cb_threads;
  int ab_threads;
} kernel_control_t;

extern kernel_control_t kernel_control;
