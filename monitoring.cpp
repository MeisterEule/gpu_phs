#include <sys/time.h>

#include "monitoring.h"

double gpu_timers[5] = {0, 0, 0, 0, 0};

FILE *logfl[2] = {NULL, NULL};

double mysecond () {
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp,&tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void init_logfiles (char *filename_input, char *filename_cuda) {
   logfl[LOG_INPUT] = fopen (filename_input, "w+");
   logfl[LOG_CUDA] = fopen (filename_cuda, "w+");
}

void final_logfiles () {
   for (int i = 0; i < 2; i++) {
      fclose(logfl[i]);
      logfl[i] = NULL;
   }
}
