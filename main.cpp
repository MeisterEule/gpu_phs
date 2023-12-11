#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <math.h>
#include <sys/time.h>

#include "phs.h"

double mysecond () {
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp,&tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int count_nevents_in_reference_file (char *ref_file, int n_momenta, int filepos) {
   int n_lines = 0;
   std::ifstream reader (ref_file);
   reader.seekg(filepos, reader.beg);
   std::string line; 

   while (getline (reader, line)) {
      n_lines++;
   }
   // Subtract the first five header lines
   int n_lines_per_batch = n_momenta + 5;
   return n_lines / n_lines_per_batch;
}

#define NHEADER 6
enum {H_NCHANNELS=0, H_NIN=1, H_NOUT=2, H_NTREES=3, H_NGROVES=4, H_NX=5};

void read_reference_header (char *ref_file, int *header_data, int *filepos) {
   std::ifstream reader (ref_file);
   std::string line; 
   std::string dummy;

   int c = 0;

   while (getline (reader, line) && c < NHEADER) {
      std::stringstream ss(line);
      ss >> dummy >> header_data[c];
      c++;
      *filepos = reader.tellg();
   }
}

void read_tree_structures (char *ref_file, int n_trees, int *filepos) {
   std::ifstream reader (ref_file);
   std::string line;
   std::string dummy;
   reader.seekg (*filepos, reader.beg);

   int counter = 0;
   int tmp;
   const int n_line_elements = 6; // 2 x daughters + children + mappings + masses + widths
   while (getline (reader, line) && counter < n_line_elements * n_trees) {
      std::stringstream ss(line);
      ss >> dummy;
      int cm = counter % n_line_elements;
      int cd = counter / n_line_elements;
      if (cm == 0) {
         for (int i = 0; i < N_PRT; i++) {
            ss >> tmp;
            daughters1[cd][i] = tmp - 1;
         }
      } else if (cm == 1) {
         for (int i = 0; i < N_PRT; i++) {
            ss >> tmp;
            daughters2[cd][i] = tmp - 1;
         }
      } else if (cm == 2) {
         for (int i = 0; i < N_PRT; i++) {
            ss >> has_children[cd][i];
         }
      } else if (cm == 3) {
         int n = 0;
         while (n < N_PRT_OUT) {
            ss >> mappings_host[cd].map_id[n];
            n++;
         }
      } else if (cm == 4) {
         int n = 0;
         while (n < N_PRT_OUT) {
            ss >> mappings_host[cd].masses[n];
            n++;
         }
      } else {
         int n = 0;
         while (n < N_PRT_OUT) {
            ss >> mappings_host[cd].widths[n];
            n++;
         }
      }
     counter++;
     *filepos = reader.tellg();
   } 
}

void read_reference_momenta (char *ref_file, int filepos, int n_momenta, int n_x, double *x,
                             int *channel_lims, phs_val_t *p) {
   std::ifstream reader (ref_file);
   std::string line; 
   int n_lines_per_batch = 5 + n_momenta; // channel + random numbers + factors + volumes + nm * Momenta + ok
   std::string *linebatch = (std::string*)malloc(n_lines_per_batch * sizeof(std::string));
   reader.seekg(filepos, reader.beg);

   int counter = 0;
   int p_counter = 0;
   std::string id;
   int channel;
   int current_channel = 1;

   int i_event = 0;
   while (getline (reader, line)) {
      int c = counter % n_lines_per_batch;
      linebatch[c] = line; 
      if (c == n_lines_per_batch - 1) {
         int i_event = counter / n_lines_per_batch;
         std::stringstream ss;
         ss.str(linebatch[0]);
         // Channel ID
         ss >> id;
         ss >> channel;
         if (channel != current_channel) {
             channel_lims[current_channel] = i_event;
             current_channel++;
         }
         ss.clear();
         ss.str(linebatch[1]);
         ss >> id;
         // Random numbers
         for (int i = 0; i < n_x; i++) {
            ss >> x[n_x * i_event + i];  
         }
         // Momenta
         for (int i = 0; i < n_momenta; i++) {
            ss.clear();
            ss.str(linebatch[2 + i]);
            for (int j = 0; j < 4; j++) {
               ss >> p[i_event].prt[i].p[j];
            }
         } 
         // Factor & Volume
         ss.clear();
         ss.str(linebatch[2 + n_momenta]); 
         ss >> p[i_event].f;
         ss.clear();
         ss.str(linebatch[2 + n_momenta + 1]); 
         ss >> p[i_event].v;
         // OKAY
         ss.clear();
         ss.str(linebatch[2 + n_momenta + 2]); 
         ss >> p[i_event].ok;
      }
      counter++;
      //printf ("DONE\n");
   }
}

#define EPSILON 0.0001

void compare_phs_gpu (FILE *fp, int n_events_val, int n_events_gen, int *channels, int n_in, int n_out, phs_val_t *pval,
                      double *prt, double *factors, double *volumes) {
   int n_events_failed = 0;
   for (int i = 0; i < n_events_gen; i++) {
      for (int n = 0; n < n_out; n++) {
         double *p = pval[i].prt[n_in+n].p;
         int nn = pow(2,n) - 1;
         if (fabs (p[0] - prt[PRT_STRIDE * i + 4 * nn + 0]) > EPSILON
          || fabs(p[1] - prt[PRT_STRIDE * i + 4 * nn + 1]) > EPSILON
          || fabs(p[2] - prt[PRT_STRIDE * i + 4 * nn + 2]) > EPSILON
          || fabs(p[3] - prt[PRT_STRIDE * i + 4 * nn + 3]) > EPSILON) {
            fprintf (fp, "Error in p%d: (event: %d, channel: %d):\n", n_in + n + 1, i, channels[i]);
            fprintf (fp, "Validation: %lf %lf %lf %lf\n", p[0], p[1], p[2], p[3]);
            fprintf (fp, "Generated:  %lf %lf %lf %lf\n", prt[PRT_STRIDE * i + 4 * nn + 0], prt[PRT_STRIDE * i + 4 * nn + 1],
                                                     prt[PRT_STRIDE * i + 4 * nn + 2], prt[PRT_STRIDE * i + 4 * nn + 3]);
            n_events_failed++;
         }

      }
      if (fabs (pval[i].f - factors[N_PRT * i + ROOT_BRANCH]) > EPSILON) {
         fprintf (fp, "Error in factor (%d): Validation: %lf, Generated: %lf\n", i, pval[i].f, factors[N_PRT * i + ROOT_BRANCH]);
         n_events_failed++;
      }

      if (fabs (pval[i].v - volumes[N_PRT * i + ROOT_BRANCH]) > EPSILON) {
         fprintf (fp, "Error in volume (%d): Validation: %lf, Generated: %lf\n", i, pval[i].v, volumes[N_PRT * i + ROOT_BRANCH]);
         n_events_failed++;
      }
   } 
   fprintf (fp, "Failed events with EPSILON = %lf: %d / %d\n", EPSILON, n_events_failed, n_events_gen);
}

void compare_phs_cpu (FILE *fp, int n_events_val, int n_events_gen, int *channels, int n_in, int n_out, phs_val_t *pval,
                      phs_prt_t *prt, double *factors, double *volumes) {
   int n_events_failed = 0;
   //for (int i = 0; i < n_events_val && ok; i++) {
   for (int i = 0; i < n_events_val; i++) {
      for (int n = 0; n < n_out; n++) {
         double *p = pval[i].prt[n_in+n].p;
         int nn = pow(2,n) - 1;
         if (fabs (p[0] - prt[N_PRT*i + nn].p[0]) > EPSILON
          || fabs (p[1] - prt[N_PRT*i + nn].p[1]) > EPSILON
          || fabs (p[2] - prt[N_PRT*i + nn].p[2]) > EPSILON
          || fabs (p[3] - prt[N_PRT*i + nn].p[3]) > EPSILON) {
               fprintf (fp, "Error in p%d (event: %d, channel: %d):\n", n, i, channels[i]);
               fprintf (fp, "Validation: %lf %lf %lf %lf\n", p[0], p[1], p[2], p[3]);
               fprintf (fp, "Generated:  %lf %lf %lf %lf\n", prt[N_PRT*i + nn].p[0], prt[N_PRT*i + nn].p[1],
                                                        prt[N_PRT*i + nn].p[2], prt[N_PRT*i + nn].p[3]);
               n_events_failed++;
         }

         if (fabs (pval[i].f - factors[i]) > EPSILON) {
            fprintf (fp, "Error in factor (%d): Validation: %lf, Generated: %lf\n", i, pval[i].f, factors[i]);
            n_events_failed++;
         }

         if (fabs (pval[i].v - volumes[i]) > EPSILON) {
            fprintf (fp, "Error in volume (%d): Validation: %lf, Generated: %lf\n", i, pval[i].v, volumes[i]);
            n_events_failed++;
         }
     }
   }
   fprintf (fp, "Failed events with EPSILON = %lf: %d / %d\n", EPSILON, n_events_failed, n_events_gen);
}

int main (int argc, char *argv[]) {
   if (argc < 2) {
      printf ("No reference file given!\n");
      return -1;
   }
   char *ref_file = argv[1];

   int n_prt_tot, n_prt_out;
   int filepos = 0;
   //read_reference_header (ref_file, &n_channels, &n_in, &n_out, &n_trees, &n_forests, &filepos);
   int *header_data = (int*)malloc (NHEADER * sizeof(int));
   read_reference_header (ref_file, header_data, &filepos);
   int n_channels = header_data[H_NCHANNELS];
   int n_in = header_data[H_NIN];
   int n_out = header_data[H_NOUT];
   int n_trees = header_data[H_NTREES];
   int n_forests = header_data[H_NGROVES];
   int n_x = header_data[H_NX];

   N_PRT = pow(2,n_in + n_out) - 1;
   N_PRT_OUT = pow(2, n_out) - 1;
   PRT_STRIDE = 4 * N_PRT;
   ROOT_BRANCH = N_PRT_OUT - 1;

#ifdef _VERBOSE
   printf ("n_channels: %d\n", n_channels);
   printf ("n_in: %d\n", n_in);
   printf ("n_out: %d\n", n_out);
   printf ("n_trees: %d\n", n_trees);
   printf ("nx: %d\n", n_x);
   printf ("NPRT: %d\n", N_PRT);
   printf ("NPRT_OUT: %d\n", N_PRT_OUT);
#endif

   daughters1 = (int**)malloc(n_trees * sizeof(int*));
   daughters2 = (int**)malloc(n_trees * sizeof(int*));
   has_children = (int**)malloc(n_trees * sizeof(int*));
   mappings_host = (mapping_t*)malloc(n_trees * sizeof(mapping_t));
   for (int i = 0; i < n_trees; i++) {
      daughters1[i] = (int*)malloc(N_PRT * sizeof(int));
      daughters2[i] = (int*)malloc(N_PRT * sizeof(int));
      has_children[i] = (int*)malloc(N_PRT * sizeof(int));
      mappings_host[i].map_id = (int*)malloc(N_PRT_OUT * sizeof(int));
      mappings_host[i].comp_msq = NULL;
      mappings_host[i].comp_ct = NULL;
      mappings_host[i].a = (map_constant_t*)malloc(N_PRT_OUT * sizeof(map_constant_t));
      mappings_host[i].b = (map_constant_t*)malloc(N_PRT_OUT * sizeof(map_constant_t));
      mappings_host[i].masses = (double*)malloc(N_PRT_OUT * sizeof(double));
      mappings_host[i].widths = (double*)malloc(N_PRT_OUT * sizeof(double));
   }
   read_tree_structures (ref_file, n_trees, &filepos);

#ifdef _VERBOSE
   for (int c = 0; c < n_channels; c++) {
      printf ("Channel %d: \n", c);
      printf ("daughters1: ");
      for (int i = 0; i < N_PRT; i++) {
         printf ("%d ", daughters1[c][i]);
      }
      printf ("\ndaughters2: ");
      for (int i = 0; i < N_PRT; i++) {
         printf ("%d ", daughters2[c][i]);
      }
      printf ("\nhas_children: ");
      for (int i = 0; i < N_PRT; i++) {
         printf ("%d ", has_children[c][i]);
      }
      printf ("\nmappings: ");
      for (int i = 0; i < N_PRT_OUT; i++) {
         printf ("%d ", mappings_host[c].map_id[i]);
      }
      printf ("\nmasses: ");
      for (int i = 0; i < N_PRT_OUT; i++) {
         printf ("%lf ", mappings_host[c].masses[i]);
      }
      printf ("\nwidths: ");
      for (int i = 0; i < N_PRT_OUT; i++) {
         printf ("%lf ", mappings_host[c].widths[i]);
      }
      printf ("\n");
   }
#endif

   phs_dim_t d;
   d.n_events_val = count_nevents_in_reference_file (ref_file, n_in + n_out, filepos);

#ifdef _VERBOSE
   printf ("n_events in reference file: %d\n", d.n_events_val);
   printf ("Memory required for reference momenta: %lf GiB\n", (double)(d.n_events_val * (n_in + n_out) * sizeof(phs_prt_t)) / 1024 / 1024 / 1024);
#endif

   double sqrts = 1000;
   double *x = (double*)malloc(n_x * d.n_events_val * sizeof(double));

   phs_val_t *pval = (phs_val_t*)malloc(d.n_events_val * sizeof (phs_val_t));
   for (int i = 0; i < d.n_events_val; i++) {
      pval[i].prt = (phs_prt_t*)malloc((n_in + n_out) * sizeof(phs_prt_t));
   }

   int *channel_lims = (int*)malloc((n_trees + 1) * sizeof(int));
   channel_lims[0] = 0;
   channel_lims[n_trees] = d.n_events_val;
   read_reference_momenta (ref_file, filepos, n_in + n_out, n_x, x, channel_lims, pval);
   d.n_events_gen = d.n_events_val;

   int *channels = (int*)malloc(d.n_events_gen * sizeof(int));
   int c = 0;
   for (int i = 0; i < d.n_events_gen; i++) {
      if (i == channel_lims[c+1]) c++;
      channels[i] = c;
   }


#ifdef _VERBOSE
   printf ("channel_limits: ");
   for (int i = 0; i < n_trees + 1; i++) {
      printf ("%d ", channel_lims[i]);
   }
   printf ("\n");
   printf ("n_events to generate: %d\n", d.n_events_gen);
#endif

   long long mem_gpu = count_gpu_memory_requirements (d, n_x);

   double *p = (double*)malloc(PRT_STRIDE * d.n_events_gen * sizeof(double));
   double *factors = (double*)malloc(N_PRT * d.n_events_gen * sizeof(double)); 
   double *volumes = (double*)malloc(N_PRT * d.n_events_gen * sizeof(double)); 
   int *oks = (int*)malloc(N_PRT * d.n_events_gen * sizeof(int));

   init_mapping_constants (n_trees, sqrts * sqrts, 0, sqrts * sqrts);
   init_phs_gpu(n_trees, mappings_host, sqrts * sqrts);
   double t1 = mysecond();
   gen_phs_from_x_gpu (sqrts, d, n_trees, channel_lims, n_x, x, factors, volumes, oks, p);
   double t2 = mysecond();

   FILE *fp = fopen ("compare.gpu", "w+");
   compare_phs_gpu (fp, d.n_events_val, d.n_events_gen, channels, n_in, n_out, pval, p, factors, volumes);
   fclose(fp);
   fp = NULL;
   printf ("dt: %lf sec\n", t2 - t1);

   free (p);
   free (factors);
   free (volumes);
   free (oks);

  
   phs_prt_t *prt = (phs_prt_t*)malloc(N_PRT * d.n_events_gen * sizeof(phs_prt_t));
   factors = (double*)malloc(d.n_events_gen * sizeof(double));
   volumes = (double*)malloc(d.n_events_gen * sizeof(double));

   t1 = mysecond();
   gen_phs_from_x_cpu (sqrts, d, n_x, x, channels, factors, volumes, prt);
   t2 = mysecond();

   fp = fopen ("compare.cpu", "w+");
   compare_phs_cpu (fp, d.n_events_val, d.n_events_gen, channels, n_in, n_out, pval, prt, factors, volumes);
   fclose(fp);
   fp = NULL;

   printf ("dt: %lf sec\n", t2 - t1);


   free (factors);
   free (volumes);
   free (prt);
   free(x);
   free(pval);
   //free(pval1);
   //free(pval2);
   //free(pval3);
   //free(pval4);
   //free(fval);
   //free(vval);
   return 0;
}
