#include <fstream>
#include <string>
#include <sstream>

#include "phs.h"
#include "mom_generator.h"

long long count_nevents_in_reference_file (char *ref_file, int n_momenta, int filepos) {
   long long n_lines = 0;
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

void read_tree_structures (char *ref_file, int n_trees, int n_prt, int n_prt_out, int *filepos) {
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
         for (int i = 0; i < n_prt; i++) {
            ss >> tmp;
            daughters1[cd][i] = tmp - 1;
         }
      } else if (cm == 1) {
         for (int i = 0; i < n_prt; i++) {
            ss >> tmp;
            daughters2[cd][i] = tmp - 1;
         }
      } else if (cm == 2) {
         for (int i = 0; i < n_prt; i++) {
            ss >> has_children[cd][i];
         }
      } else if (cm == 3) {
         int n = 0;
         while (n < n_prt_out) {
            ss >> mappings_host[cd].map_id[n];
            n++;
         }
      } else if (cm == 4) {
         int n = 0;
         while (n < n_prt_out) {
            ss >> mappings_host[cd].masses[n];
            n++;
         }
      } else {
         int n = 0;
         while (n < n_prt_out) {
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



