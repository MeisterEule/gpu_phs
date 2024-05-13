#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <queue>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"

#include "phs.h"
#include "file_input.h"

input_control_t input_control;

void read_input_json (const char *filename) {
   std::string line, text;
   std::ifstream in (filename);
   while (std::getline(in, line)) {
      text += line + "\n";
   }

   rapidjson::Document d;
   d.Parse (text.c_str());  

   assert (d["ref_file"].IsString());
   input_control.ref_file = strdup(d["ref_file"].GetString());

   bool verify_whizard;
   if (d.HasMember("verify")) {
      assert (d["verify"]["against"].IsString());
      verify_whizard = !strcmp(d["verify"]["against"].GetString(), "whizard");
      if (d["verify"].HasMember("epsilon")) {
         input_control.compare_tolerance = d["verify"]["epsilon"].GetDouble();
      } else {
         input_control.compare_tolerance = DEFAULT_COMPARE_EPSILON;
      }
   } else {
      verify_whizard = true;
      input_control.compare_tolerance = DEFAULT_COMPARE_EPSILON;
   }

   if (!verify_whizard) {
       if (d.HasMember("n_events") && d.HasMember("gpu_memory")) {
          printf ("%s: Both n_events and gpu_memory specified. gpu_memory takes precedence.\n", filename);
       }
       if (d.HasMember("gpu_memory")) {
          assert (d["gpu_memory"].IsInt());
          input_control.run_type = RT_INTERNAL_FIXED_MEMORY;
          input_control.gpu_memory = d["gpu_memory"].GetInt64() * 1024 * 1024;
       } else if (d.HasMember("n_events")) {
          assert (d["n_events"].IsInt());
          input_control.run_type = RT_INTERNAL_FIXED_N;
          input_control.internal_events = d["n_events"].GetInt64();
       } else {
          std::abort();
       }
       if (d.HasMember("warmup")) {
          assert (d["warmup"]["n_trials"].IsInt());
          input_control.warmup_trials = d["warmup"]["n_trials"].GetInt();
          assert (d["warmup"]["n_events"].IsInt());
          input_control.warmup_events = d["warmup"]["n_events"].GetInt64();
       } else {
          input_control.warmup_trials = 0;
          input_control.warmup_events = 0;
       }
   } else {
       input_control.run_type = RT_WHIZARD;
   }

   if (d.HasMember("msq")) {
      assert (d["msq"]["threads"].IsInt());
      input_control.msq_threads = d["msq"]["threads"].GetInt();
   } else {
      input_control.msq_threads = NTHREADS_DEFAULT;
   } 

   if (d.HasMember("create_boosts")) {
      assert (d["create_boosts"]["threads"].IsInt());
      input_control.cb_threads = d["create_boosts"]["threads"].GetInt();
   } else {
      input_control.cb_threads = NTHREADS_DEFAULT;
   } 

   if (d.HasMember("apply_boosts")) {
      assert (d["apply_boosts"]["threads"].IsInt());
      input_control.ab_threads = d["apply_boosts"]["threads"].GetInt();
   } else {
      input_control.ab_threads = NTHREADS_DEFAULT;
   } 

   if (d.HasMember("check_cpu")) {
      assert (d["check_cpu"].IsBool());
      input_control.check_cpu = d["check_cpu"].GetBool();
   } else {
      input_control.check_cpu = false;
   }

}

size_t count_nevents_in_reference_file (const char *ref_file, int n_momenta, int filepos) {
   size_t n_lines = 0;
   std::ifstream reader (ref_file);
   reader.seekg(filepos, reader.beg);
   std::string line; 

   while (getline (reader, line)) {
      n_lines++;
   }
   // Each phase space point has, regardless of process, five additional lines:
   // factor, volume, channel, ???, and random numbers. 
   int n_lines_per_batch = n_momenta + 5;
   return n_lines / n_lines_per_batch;
}

void read_reference_header (const char *ref_file, int *header_data, int *filepos) {
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

void read_tree_structures (const char *ref_file, int n_trees, int n_prt, int n_prt_out, int n_external, int *filepos) {
   std::ifstream reader (ref_file);
   std::string line;
   std::string dummy;
   reader.seekg (*filepos, reader.beg);

   int counter = 0;
   int tmp;
   const int n_line_elements = 7; // 2 x daughters + children + friends + mappings + map_masses + map_widths
   while (counter < n_line_elements * n_trees) {
      getline (reader, line);
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
         ss >> contains_friends[cd];
      } else if (cm == 4) {
         int n = 0;
         while (n < n_prt_out) {
            ss >> mappings_host[cd].map_id[n];
            n++;
         }
      } else if (cm == 5) {
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
   }
   // flv_masses + flv_widths
   getline (reader, line);
   std::stringstream ss(line);
   ss >> dummy;
   for (int i = 0; i < n_external; i++) {
      ss >> flv_masses[i];
   }
   getline (reader, line);
   ss.clear();
   ss.str(line);
   ss >> dummy;
   for (int i = 0; i < n_external; i++) {
      ss >> flv_widths[i];
   }
   *filepos = reader.tellg();
}

void read_reference_momenta (const char *ref_file, int filepos,
                             int n_momenta, int n_x, double *x,
                             int *channel_lims, phs_val_t *p) {
   std::ifstream reader (ref_file);
   std::string line; 
   int n_lines_per_batch = 5 + n_momenta; // channel + random numbers + factors + volumes + nm * Momenta + ok
   std::queue<std::string> linebatch;
   reader.seekg(filepos, reader.beg);

   int counter = 0;
   int p_counter = 0;
   std::string id;
   int channel;
   int current_channel = 1;

   int i_event = 0;
   while (getline (reader, line)) {
      int c = counter % n_lines_per_batch;
      linebatch.push(line);
      if (c == n_lines_per_batch - 1) {
         int i_event = counter / n_lines_per_batch;
         std::stringstream ss;
         ss.str(linebatch.front());
         linebatch.pop();
         ss >> id;
         ss >> channel;
         if (channel != current_channel) {
             channel_lims[current_channel] = i_event;
             current_channel++;
         }
         ss.clear();
         ss.str(linebatch.front());
         linebatch.pop();
         ss >> id;
         // Random numbers
         for (int i = 0; i < n_x; i++) {
            ss >> x[n_x * i_event + i];  
         }
         // Momenta
         for (int i = 0; i < n_momenta; i++) {
            ss.clear();
            ss.str(linebatch.front());
            linebatch.pop();
 
            for (int j = 0; j < 4; j++) {
               ss >> p[i_event].prt[i].p[j];
            }
         } 
         // Factor & Volume
         ss.clear();
         ss.str(linebatch.front()); 
         linebatch.pop();
         ss >> p[i_event].f;
         ss.clear();
         ss.str(linebatch.front()); 
         linebatch.pop();
         ss >> p[i_event].v;
         // OKAY
         ss.clear();
         ss.str(linebatch.front()); 
         linebatch.pop();
         ss >> p[i_event].ok;
      }
      counter++;
   }
}



