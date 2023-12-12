#ifndef MOM_GENERATOR_H
#define MOM_GENERATOR_H

#include "phs.h"

#define NHEADER 6
enum {H_NCHANNELS=0, H_NIN=1, H_NOUT=2, H_NTREES=3, H_NGROVES=4, H_NX=5};

int count_nevents_in_reference_file (char *ref_file, int n_moment, int filepos);
void read_reference_header (char *ref_file, int *header_data, int *filepos);
void read_tree_structures (char *ref_file, int n_trees, int n_prt, int n_prt_out, int *filepos);
void read_reference_momenta (char *ref_file, int filepos, int n_momenta, int n_x,
                             double *x, int *channel_lims, phs_val_t *p);

#endif
