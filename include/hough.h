#ifndef _MY_HOUGH_LINES_
#define _MY_HOUGH_LINES_

#include "mymat.h"

#define SCAN_AREA 50
#define SCAN_DELTA_THETA 0.05
#define SCAN_DELTA_RU 5
#define SCAN_AREA_THO SCAN_AREA/4
#define MIN_POINTS SCAN_AREA*2

keypoint hough(mat_c* src);
int scan_subarea(mat_c* src, int si, int sj, float* ru, float* theta);
int scan_allarea(mat_c* src, int si, int sj, float ru, float theta, int isdelete);
#endif 
