#ifndef _MY_CANNY_EDGE_DETECTION_
#define _MY_CANNY_EDGE_DETECTION_
#include "mymat.h"
#include <math.h>

//Scale floating point magnitudes and angles to 8 bits
#define ORI_SCALE 10.0
#define MAG_SCALE 20.0

//parameters
#define LOW_THOD 120
#define HIGH_THOD 180

//Biggest possible filter mask
#define MAX_MASK_SIZE 10

int range(mat_c* x, int i, int j);

float norm(float x, float y);

void canny(float s, mat_c* im, mat_c* mag, mat_c* ori);

//Gaussian
float gauss(float x, float sigma);

float meanGauss(float x, float sigma);

//First derivative of Gaussian
float dGauss(float x, float sigma);

/*HYSTERESIS thersholding of edge pixels. 
 *Starting at pixels with a value greater than ther HIGH threshold,
 *trace a connected sequence of pixels that have a value greater than
 *the LOW threhsold.*/
void hysteresis(int high, int low, mat_c* im, mat_c* mag, mat_c* oriim);

/* TRACE - recursively trace edge pixels that have a threshold > the low
 * edge threshold, continuing from the pixel at (i,j). */
int trace(int i, int j, int low, mat_c* im, mat_c* mag, mat_c* ori);

void seperable_convolution(mat_c* im, float * gau, int width, mat_c *smx, mat_c *smy);

void dxy_seperable_convolution(mat_c* im, int nr, int nc, float *gau, int width, mat_c *sm, int which);

void nonmax_suppress(mat_c *dx, mat_c *dy, int nr, int nc, mat_c* mag, mat_c* ori);

void estimate_thresh(mat_c* mag, int *hi, int *low);

int canny_edge(mat_c* im);

#endif
