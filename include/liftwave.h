#ifndef _MY_LIFTING_WAVELETS_
#define _MY_LIFTING_WAVELETS_

#include "mymat.h"

#define MAX_LOOP_COUNT 10
#define DECREASE_RATE 0.1
#define LIFT_EPSILON_INIT 0.33


int liftwave_low_2d(mat_c* src, mat_c* dst);
int liftwave_high_2d(mat_c* src, mat_c* dst);
#endif
