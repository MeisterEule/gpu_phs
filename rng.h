#ifndef RNG_H
#define RNG_H

void init_rng (int n_channels, int n_dim); 
void rng_generate (int channel, long long n_events, int n_dim, double *x);
void update_weights (int n_dim, int n_channels, long long n_events,
                     int *channels, double *x, bool *ks);

#endif
