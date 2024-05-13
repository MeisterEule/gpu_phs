#ifndef FILE_INPUT_H
#define FILE_INPUT_H

#include "phs.h"

#define NTHREADS_DEFAULT 512
#define BYTES_PER_GB 1073741824
#define BYTES_PER_MB    1048576

#define DEFAULT_COMPARE_EPSILON 0.001

enum {RT_INTERNAL_FIXED_N, RT_INTERNAL_FIXED_MEMORY, RT_WHIZARD};

typedef struct {
  const char *ref_file;
  int run_type;  
  size_t warmup_trials;
  int warmup_events;
  size_t internal_events;
  size_t gpu_memory;
  int msq_threads;
  int cb_threads;
  int ab_threads; 
  bool check_cpu;
  double compare_tolerance;
} input_control_t;

extern input_control_t input_control;

#define NHEADER 6
enum {H_NCHANNELS=0, H_NIN=1, H_NOUT=2, H_NTREES=3, H_NGROVES=4, H_NX=5};

void read_input_json (const char *filename);

size_t count_nevents_in_reference_file (const char *ref_file, int n_moment, int filepos);
int read_reference_header (const char *ref_file, int *header_data, int *filepos);
void read_tree_structures (const char *ref_file, int n_trees, int n_prt, int n_prt_out, int n_external, int *filepos);
void read_reference_momenta (const char *ref_file, int filepos, int n_momenta, int n_x,
                             double *x, int *channel_lims, phs_val_t *p);

#endif
