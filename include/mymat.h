#ifndef MY_MAT_H_
#define MY_MAT_H_

#include <math.h>

#define MAT_UC2D 0
#define MAT_F2D 1
#define MAT_I2D 2


#define PI 3.1415926535
#define SIGMA sqrt(3)
#define GAUSSKERN 3.5

typedef struct {
  int type;
  int step;          // bytes for a line
  int rows, cols;    // size of an image
  union
  {
    unsigned char** ptr;
    float** f2d;
    int** i2d;
  } data;
}mat_c;

//keypoint:Lists of keypoints are linked by the "next" field.  
typedef struct KeypointSt   
{  
  float row, col; /* size of im and positions of points*/  
  float sx,sy;    /* positions of keypoints*/  
  int octave,level;/*level and octave of keypoint*/  
   
  float scale, ori,mag; /*sigma,orientation (range [-PI,PI])，and scale*/  
  float *descrip;       /*descriptor：128 or 32*/
  
  struct KeypointSt *match;/* Pointer to matched keypoint in list. */ 
  struct KeypointSt *next;/* Pointer to next keypoint in list. */  
} *keypoint;  

mat_c* malloc_mat(int rows, int cols, int channels, int type);
mat_c* copy_mat(mat_c* mat);
int convtp_mat(mat_c* src, mat_c* dst);
int f2uc_mat(mat_c* src, mat_c* dst);
void free_mat(mat_c* mat);
int rgb2gray(mat_c* src, mat_c* dst);

int sub_mat(const mat_c* src1, const mat_c* src2, mat_c* dst); //src1-src2
void zero_mat(const mat_c* mat);
int conv_scale_mat(const mat_c* mat, float scale);

mat_c* half_size_image(mat_c* im);     //half size of image
mat_c* doubleSizeImage(mat_c* im);   //double the image
mat_c* doubleSizeImage2(mat_c* im);  //double the image 
float get_pixel_bi(mat_c* im, float col, float row);//BI
void normalize_vec(float* vec, int dim);//normalize    
mat_c* gaussian_kernel_2d(float sigma);  //2d gaussian kernel  
void normalize_mat(mat_c* mat) ;        //normalize  
float* gaussian_kernel_1d(float sigma, int dim); //1d gaussian kernel  
float get_vec_norm( float* vec, int dim );
float get_e_distence( float* vec1, float* vec2 ,int dim );
  
//convolve
float convolve_loc_width(float* kernel, int dim, mat_c* src, int x, int y);   
void convolve_1d_width(float* kern, int dim, mat_c* src, mat_c* dst);      
float convolve_loc_height(float* kernel, int dim, mat_c* src, int x, int y);
void convolve_1d_height(float* kern, int dim, mat_c* src, mat_c* dst);       
int blur_image(mat_c* src, mat_c* dst, float sigma);

//keypoints
void free_keypoint(keypoint *k);
keypoint copy_keypoint(keypoint p);

#ifdef CV
#include "cv.h"
#include "highgui.h"
#include "ml.h"
#include "sift.h"

int imread_cv(cv::Mat, mat_c*);
cv::Mat imwrite_cv(const mat_c*);
void display_keypoint_location(cv::Mat image, keypoint keypoints);
void display_keypoint_line(cv::Mat image, keypoint keypoints);
#endif

#endif
