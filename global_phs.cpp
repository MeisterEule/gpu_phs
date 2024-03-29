#include <stdlib.h>

#include "mappings.h"

int N_EXT_IN = 0;
int N_EXT_OUT = 0;
int N_EXT_TOT = 0;

int N_PRT = 0;
int N_PRT_IN = 0;
int N_PRT_OUT = 0;
int PRT_STRIDE = 0;
int ROOT_BRANCH = 0;
int N_BRANCHES = 0;
int N_BRANCHES_INTERNAL = 0;
int N_MSQ = 0;
int N_BOOSTS = 0;
int N_LAMBDA_IN = 0;
int N_LAMBDA_OUT = 0; 

mapping_t *mappings_host = NULL;
