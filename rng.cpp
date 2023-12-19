#include <random>
#include <array>
#include <cstring>

#define NBINS 10

typedef struct {
   double w[NBINS];
} weight_t;

typedef struct {
   double intervals[NBINS+1];
   double weights[NBINS];
} x_dim_t;

//typedef struct {
//   double *x;
//} x_channel_t;

x_dim_t **x_dim;

void init_rng (int n_channels, int n_dim) {
   x_dim = (x_dim_t**)malloc(n_channels * sizeof(x_dim_t*));
   for (int c = 0; c < n_channels; c++) {
      x_dim[c] = (x_dim_t*)malloc(n_dim * sizeof(x_dim_t));
      for (int i = 0; i < n_dim; i++) {
         for (int b = 0; b < NBINS; b++) {
            x_dim[c][i].intervals[b] = 1.0/NBINS * b;
            x_dim[c][i].weights[b] = 1.0/NBINS;
         }
         x_dim[c][i].intervals[NBINS] = 1.0;
      }
   }

   //printf ("Init weights: \n");
   //for (int c = 0; c < 1; c++) {
   //   printf ("Channel: %d\n", c);
   //   for (int d = 0; d < n_dim; d++) {
   //      printf ("  dim: %d\n", d);
   //      for (int b = 0; b < NBINS; b++) {
   //         printf ("    %lf ", x_dim[c][d].weights[b]);
   //      }
   //      printf ("\n");
   //   }
   //}

}


void update_weights (int n_dim, int n_channels, int n_events, int *channels, double *x, bool *oks) {

   int *n_tot = (int*)malloc(n_channels * sizeof(int));
   for (int c = 0; c < n_channels; c++) {
      n_tot[c] = 0;
      for (int d = 0; d < n_dim; d++) {
         for (int b = 0; b < NBINS; b++) {
            x_dim[c][d].weights[b] = 0;
         }
      }
   }


   for (int i = 0; i < n_events; i++) {
      int c = channels[i];
      if (oks[i]) {
         for (int j = 0; j < n_dim; j++) {
            int idx = (int)(x[n_dim * i + j] * NBINS); 
            x_dim[c][j].weights[idx]++;
         }
         n_tot[c]++;
      }
   }

   for (int c = 0; c < n_channels; c++) {
      for (int d = 0; d < n_dim; d++) {
         for (int b = 0; b < NBINS; b++) {
            x_dim[c][d].weights[b] /= n_tot[c];
         }
      }
   }

   //printf ("New weights: \n");
   //for (int c = 0; c < 1; c++) {
   //   printf ("Channel: %d\n", c);
   //   for (int d = 0; d < n_dim; d++) {
   //      printf ("  dim: %d\n", d);
   //      for (int b = 0; b < NBINS; b++) {
   //         printf ("    %lf ", x_dim[c][d].weights[b]);
   //      }
   //      printf ("\n");
   //   }
   //}

   free(n_tot);

   
   //for (int c = 0; c < n_channels; c++) {
   //   for (int d = 0; 
   //}
   //for (int b = 0; b < NBINS; b++) {
   //   for (int j = 0; j < dim; j++) {
   //      weights[channel][j].w[b] = 0;
   //   }
   //} 
   //for (int i = 0; i < n_events; i++) {
   //   if (oks[i]) {
   //      for (int j = 0; j < dim; j++) {
   //         int idx = (int)x[dim * i + j] * NBINS; 
   //         weights[channel][j].w[idx]++;
   //      }
   //   }
   //}
   //for (int j = 0; j < dim; j++) {
   //   for (int b = 0; b < NBINS; b++) { 
   //      weights[channel][j].w[b] /= n_events;
   //   }
   //}
}

void rng_generate (int n_channels, int n_dim, int n_events_per_channel, double *x) {
   std::default_random_engine generator;
   //std::array<double,6> intervals {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
   //double intervals[6] = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
   //std::array<double,5> weights {0.25, 0.1, 0.3, 0.1, 0.25};
   //std::piecewise_constant_distribution<double> dist (intervals.begin(), intervals.end(), weights.begin());
   //printf ("Generate: %d %d %d\n", n_channels, n_dim, n_events);
   for (int c = 0; c < n_channels; c++) { 
      for (int d = 0; d < n_dim; d++) {
         std::piecewise_constant_distribution<double> dist (std::begin(x_dim[c][d].intervals),
                                                            std::end(x_dim[c][d].intervals),
                                                            std::begin(x_dim[c][d].weights));

         //double *foo = (double*)malloc(n_events * n_dim * sizeof(double));
         //for (int i = 0; i < n_events * n_dim; i++) {
         //   foo[i] = dist(generator);
         //}
         for (int i = 0; i < n_events_per_channel; i++) {
            //printf ("FILL: %d %d %d %d: %lf\n", c, i, d, c * n_dim * i + d, xx);
            //fflush(stdout);
            //x[c * i + d] = dist(generator);
            x[c * n_events_per_channel * n_dim + n_dim * i + d] = dist(generator);
         }
      }
   }

      //int *count = (int*)malloc(NBINS * sizeof(int)); 
      //memset (count, 0, NBINS * sizeof(int));
      //for (int i = 0; i < n_events * n_dim; i++) {
      //   double x = foo[i];
      //   int j;
      //   for (j = 0; j < NBINS; j++) {
      //      if (x > x_dim[channel][0].intervals[j] && x < x_dim[channel][0].intervals[j+1]) break;
      //   } 
      //   count[j]++;
      //}
      //for (int i = 0; i < NBINS; i++) {
      //   printf ("%d: %d %lf\n", i, count[i], (double)count[i] / (n_events * n_dim));
      //}
}

