#ifndef RNG_H
#define RNG_H

void init_rng (int n_channels, int n_dim); 
void rng_generate (int channel, int n_events, int n_ix, double *x);
void update_weights (int n_dim, int n_channels, int n_events, int *channels, double *x, bool *ks);

#endif
