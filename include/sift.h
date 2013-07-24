#ifndef _MY_MAT_SIFT_
#define _MY_MAT_SIFT_

#include "mymat.h"

#define NUMSIZE 2  
//Sigma of base image -- See D.L.'s paper.  
#define INITSIGMA 0.5  

//Number of scales per octave.  See D.L.'s paper.  
#define SCALESPEROCTAVE 2  
#define MAXOCTAVES 4  
  
#define CONTRAST_THRESHOLD   0.02  
#define CURVATURE_THRESHOLD  10.0  
#define DOUBLE_BASE_IMAGE_SIZE 0  
#define PEAK_REL_THRESH 0.8  
#define LEN 128     
#define MATCH_THRESHOLD  0.4  

#define GridSpacing 4  

//Data structure for a float image.  
typedef struct ImageSt 
{   
 float levelsigma;  
 int levelsigmalength;  
 float absolute_sigma;  
 mat_c *level;         
} image_levels;  
  
typedef struct ImageSt1  /*octives*/  
{   
 int row, col;          //Dimensions of image.   
 float subsample;  
 image_levels *octave;                
} image_octaves;  

mat_c *scale_init_image(mat_c* im);

image_octaves* build_gaussian_octaves(mat_c* image);

int detect_keypoint(int numoctaves, image_octaves *gaussianPyr);

void compute_grad_direcand_mag(int numoctaves, image_octaves *gaussianPyr);
int find_closest_rotation_bin (int binCount, float angle);  
void average_weak_bins (double* hist, int binCount);
bool interpolate_orientation (double left, double middle,double right, double *degreeCorrection, double *peakValue);  
void assign_the_main_orientation(int numoctaves, image_octaves *gaussianPyr,image_octaves *mag_pyr,image_octaves *grad_pyr);  

void extract_feature_descriptors(int numoctaves, image_octaves *gaussianPyr);

keypoint sift_desc(mat_c* src);
int sift_match(keypoint src1, keypoint src2);



#endif
