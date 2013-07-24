#include <stdio.h>
#include "mymat.h"


mat_c* malloc_mat(int rows, int cols, int channels, int type)
{
  int i;
  mat_c* ptr;
  ptr = (mat_c*)malloc(sizeof(mat_c));
  switch(type)
  {
  case MAT_UC2D:
    ptr->data.ptr = (unsigned char**)malloc(rows*sizeof(unsigned char*));
    for(i=0;i<rows;i++)
    {
      ptr->data.ptr[i] = (unsigned char*)malloc(cols*channels*sizeof(unsigned char));
    }
    break;
    
  case MAT_F2D:
    ptr->data.f2d = (float**)malloc(rows*sizeof(float*));
    for(i=0;i<rows;i++)
    {
      ptr->data.f2d[i] = (float*)malloc(cols*channels*sizeof(float));
    }
    break;
    
  default:
    break;

  }

  ptr->type = type;
  ptr->step = cols*channels;
  ptr->rows = rows;
  ptr->cols = cols;
  
  return ptr;
}

mat_c* copy_mat(mat_c* mat)
{
  int i,j;
  mat_c* ptr;
  ptr = (mat_c*)malloc(sizeof(mat_c));
  switch(mat->type)
  {
  case MAT_UC2D:
    ptr->data.ptr = (unsigned char**)malloc(mat->rows*sizeof(unsigned char*));
    for(i=0;i<mat->rows;i++)
    {
      ptr->data.ptr[i] = (unsigned char*)malloc(mat->step*sizeof(unsigned char));
      for(j=0;j<mat->step;j++)
      {
	ptr->data.ptr[i][j] = mat->data.ptr[i][j];
      }
    }
    break;
    
  case MAT_F2D:
    ptr->data.f2d = (float**)malloc(mat->rows*sizeof(float*));
    for(i=0;i<mat->rows;i++)
    {
      ptr->data.f2d[i] = (float*)malloc(mat->step*sizeof(float));
      for(j=0;j<mat->step;j++)
      {
	ptr->data.f2d[i][j] = mat->data.f2d[i][j];
      }
    }
    break;
    
  default:
    break;

  }

  ptr->type = mat->type;
  ptr->step = mat->step;
  ptr->rows = mat->rows;
  ptr->cols = mat->cols;
  
  return ptr;
}

int convtp_mat(mat_c* src, mat_c* dst)
{
    int i,j;
    float temp;

  if(NULL == dst || src->rows != dst->rows
     || src->step != dst->step || src->cols != dst->cols)
  {
    return 0;
  }
  switch(dst->type)
  {
  case MAT_UC2D:
    switch(src->type)
    {
    case MAT_UC2D:
      for(j=0; j<src->rows; j++)
      {
	for(i=0; i<src->step; i++)
	{
	  dst->data.ptr[j][i] = src->data.ptr[j][i];
	}
      }
      break;

    case MAT_F2D:
      for(j=0; j<src->rows; j++)
      {
	for(i=0; i<src->step; i++)
	{
	  temp = src->data.f2d[j][i];
	  if(temp<0)
	  {
	    dst->data.ptr[j][i] = 0;
	  }
	  else if(temp>255)
	  {
	    dst->data.ptr[j][i] = 255;
	  }
	  else
	  {
	    dst->data.ptr[j][i] = src->data.f2d[j][i];
	  }
	}
      }
      break;
      
    default:
      break;
    }
    break;
    ////////
  case MAT_F2D:
    switch(src->type)
    {
    case MAT_UC2D:
      for(j=0; j<src->rows; j++)
      {
	for(i=0; i<src->step; i++)
	{
	  dst->data.f2d[j][i] = src->data.ptr[j][i];
	}
      }
      break;

    case MAT_F2D:
      for(j=0; j<src->rows; j++)
      {
	for(i=0; i<src->step; i++)
	{
	  dst->data.f2d[j][i] = src->data.f2d[j][i];
	}
      }
      break;
      
    default:
      break;
    }  
    break;
    ////////
  default:
    break;
  }
  
  return 1;
}

int f2uc_mat(mat_c* src, mat_c* dst)
{
  int i,j;
  float min, max;

  if(NULL == src || MAT_F2D != src->type || NULL == dst || MAT_UC2D != dst->type
     || src->step != dst->step || src->cols != dst->cols)
  {
    return 0;
  }

  min = src->data.f2d[0][0];
  max = src->data.f2d[0][0];
  for(j=0; j<src->rows; j++)
  {
    for(i=0; i<src->step; i++)
    {
      if(src->data.f2d[j][i]>max)
      {
	max = src->data.f2d[j][i];
      }
      else
      {
	min = src->data.f2d[j][i];
      }
    }
  }

  printf("min: %f, max: %f\n",min, max);
  
  for(j=0; j<src->rows; j++)
  {
    for(i=0; i<src->step; i++)
    {
      dst->data.ptr[j][i] = (unsigned char)(src->data.f2d[j][i]-min)/(max-min)*255.0f;
    }
  }

  
}

void free_mat(mat_c* mat)
{
  int i;
  
  if(NULL == mat)
  {
    return;
  }
  
  switch(mat->type)
  {
    case MAT_UC2D:
      for(i=0;i<mat->rows;i++)
      {
	if(NULL != mat->data.ptr[i])
	{
	  free(mat->data.ptr[i]);
	}
      }

      free(mat->data.ptr);
      break;
    
  case MAT_F2D:
    for(i=0;i<mat->rows;i++)
    {
      if(NULL != mat->data.f2d[i])
      {
	free(mat->data.f2d[i]);
      }
    }

    free(mat->data.f2d);
    break;
    
  default:
    break;
  }
  
  free(mat);
  mat = NULL;
 
}

int rgb2gray(mat_c* src, mat_c* dst)
{
  int nx,ny;

  if(NULL == src && NULL == dst && 3 != src->step/src->rows && 1 != dst->step/dst->rows)
  {
    return 0;
  }

  for(ny=0; ny<dst->rows; ny++)
  {
    for(nx=0;nx<dst->cols;nx++)
    {
      dst->data.ptr[ny][nx] = (src->data.ptr[ny][3*nx] + src->data.ptr[ny][3*nx+1] + src->data.ptr[ny][3*nx+2])/3;
    }
  }
  return 1;
}

int sub_mat(const mat_c* src1, const mat_c* src2, mat_c* dst)
{
  int i,j;

  if(NULL == src1 && NULL == src2 && src1->type != src2->type 
     && src1->rows != src2->rows && src1->step != src2->step 
     && src1->cols != src2->cols)
  {
    return 0;
  }
  if(NULL == dst && src1->type != dst->type && src1->rows != dst->rows
     && src1->step != dst->step && src1->cols != dst->cols)
  {
    return 0;
  }
  switch(src1->type)
  {
  case MAT_UC2D:
    for(j=0; j<src1->rows; j++)
    {
      for(i=0; i<src1->step; i++)
      {
	dst->data.ptr[j][i] = src1->data.ptr[j][i] - src2->data.ptr[j][i];
      }
    }
    break;

  case MAT_F2D:
    for(j=0; j<src1->rows; j++)
    {
      for(i=0; i<src1->step; i++)
      {
	dst->data.f2d[j][i] = src1->data.f2d[j][i] - src2->data.f2d[j][i];
      }
    }
    break;

  default:
    break;
  }
  
  return 1;
}

int conv_scale_mat(const mat_c* mat, float scale)
{
  int i,j;
  if(NULL == mat && 0.0f == scale && MAT_F2D != mat->type)
  {
    return 0;
  }

 for(j=0; j<mat->rows; j++)
 {
   for(i=0; i<mat->step; i++)
   {
     mat->data.f2d[j][i] *= scale;
   }
 }
 return 1;
}
    
void zero_mat(const mat_c* mat)
{
  int i,j;
  
  switch(mat->type)
  {
  case MAT_UC2D:
    for(j=0; j<mat->rows; j++)
    {
      for(i=0; i<mat->step; i++)
      {
	mat->data.ptr[j][i] = 0;
      }
    }
    break;

  case MAT_F2D:
    for(j=0; j<mat->rows; j++)
    {
      for(i=0; i<mat->step; i++)
      {
	mat->data.f2d[j][i] = 0.0f;
      }
    }
    break;

  default:
    break;
  }
}

mat_c* half_size_image(mat_c* im)   
{  
  unsigned int i,j;  
  int w = im->cols/2;  
  int h = im->rows/2;   
  mat_c *imnew = malloc_mat(h, w, 1, MAT_F2D);  
  
#define Im(ROW,COL) (im->data.f2d[(ROW)][(COL)])  
#define Imnew(ROW,COL) (imnew->data.f2d[(ROW)][(COL)])  

  for ( j = 0; j < h; j++)
  { 
    for ( i = 0; i < w; i++)
    {   
      Imnew(j,i)=Im(j*2, i*2);
    }
  }  
  return imnew;  
}

mat_c* doubleSizeImage(mat_c* im)
{
  unsigned int i,j;  
  int w = im->cols*2;  
  int h = im->rows*2;   
  mat_c *imnew = malloc_mat(h, w, 1, MAT_F2D);
   
#define Im(ROW,COL) (im->data.f2d[(ROW)][(COL)])  
#define Imnew(ROW,COL) (imnew->data.f2d[(ROW)][(COL)])   
   
  for ( j = 0; j < h; j++)   
  {
    for ( i = 0; i < w; i++)
    {   
      Imnew(j,i)=Im(j/2, i/2);
    }
  }  
    
  return imnew;  
}

mat_c* doubleSizeImage2(mat_c* im)
{
  unsigned int i,j;  
  int w = im->cols*2;  
  int h = im->rows*2;   
  mat_c *imnew = malloc_mat(h, w, 1, MAT_F2D);
   
#define Im(ROW,COL) (im->data.f2d[(ROW)][(COL)])  
#define Imnew(ROW,COL) (imnew->data.f2d[(ROW)][(COL)])     
   
  // fill every pixel so we don't have to worry about skipping pixels later  
  for ( j = 0; j < h; j++)   
    {  
      for ( i = 0; i < w; i++)   
	{  
	  Imnew(j,i)=Im(j/2, i/2);  
	}  
    }  
  /* 
     A B C 
     E F G 
     H I J 
     pixels A C H J are pixels from original image 
     pixels B E G I F are interpolated pixels 
  */  
  // interpolate pixels B and I  
  for ( j = 0; j < h; j += 2)  
  {
    for ( i = 1; i < w - 1; i += 2)
    {  
      Imnew(j,i)=0.5*(Im(j/2, i/2)+Im(j/2, i/2+1));
    }
  }  
  // interpolate pixels E and G  
  for ( j = 1; j < h - 1; j += 2)  
  {
    for ( i = 0; i < w; i += 2)
    {  
      Imnew(j,i)=0.5*(Im(j/2, i/2)+Im(j/2+1, i/2));
    }
  }  
  // interpolate pixel F  
  for ( j = 1; j < h - 1; j += 2)  
  {
    for ( i = 1; i < w - 1; i += 2)
    {  
      Imnew(j,i)=0.25*(Im(j/2, i/2)+Im(j/2+1, i/2)+Im(j/2, i/2+1)+Im(j/2+1, i/2+1));
    }
  }  
  return imnew;  
}

float get_pixel_bi(mat_c* im, float col, float row)
{
  int irow, icol;  
  float rfrac, cfrac;  
  float row1 = 0, row2 = 0;  
  int width=im->cols;  
  int height=im->rows;  
#define ImMat(ROW,COL) (im->data.f2d[(ROW)][(COL)])  
   
  irow = (int) row;  
  icol = (int) col;  
   
  if (irow < 0 || irow >= height  
      || icol < 0 || icol >= width)
  {
    return 0;
  }  
  if (row > height - 1)  
  {
    row = height - 1;
  }  
  if (col > width - 1)  
  {
    col = width - 1;
  }  
  rfrac = 1.0 - (row - (float) irow);  
  cfrac = 1.0 - (col - (float) icol);  
  if (cfrac < 1)   
  {  
    row1 = cfrac * ImMat(irow,icol) + (1.0 - cfrac) * ImMat(irow,icol+1);  
  }   
  else   
  {  
    row1 = ImMat(irow,icol);  
  }  
  if (rfrac < 1)   
  {  
    if (cfrac < 1)   
    {  
      row2 = cfrac * ImMat(irow+1,icol) + (1.0 - cfrac) * ImMat(irow+1,icol+1);  
    } 
    else   
    {  
      row2 = ImMat(irow+1,icol);  
    }  
  }  
  return rfrac * row1 + (1.0 - rfrac) * row2;  
}

void normalize_vec(float* vec, int dim)
{
  unsigned int i;  
  float sum = 0;  
  for ( i = 0; i < dim; i++)  
  {
    sum += vec[i];
  }  
  for ( i = 0; i < dim; i++)
  {  
    vec[i] /= sum;
  }  
}

void normalize_mat(mat_c* mat)
{
#define Mat(ROW,COL) (mat->data.f2d[(ROW)][(COL)])  
  float sum = 0;  
  unsigned int i,j; 
   
  for (j = 0; j < mat->rows; j++)  
  {
    for (i = 0; i < mat->cols; i++)   
    {
      sum += Mat(j,i);
    }  
  }
  for ( j = 0; j < mat->rows; j++)
  {   
    for (i = 0; i < mat->rows; i++)
    {
      Mat(j,i) /= sum;
    }
  } 
}

float get_vec_norm( float* vec, int dim )  
{  
  float sum=0.0;  
  unsigned int i;
  for (i=0;i<dim;i++)
  {  
    sum+=vec[i]*vec[i];
  }  
  return sqrt(sum);  
}  

float get_e_distence( float* vec1, float* vec2 ,int dim )
{
  float sum=0.0;  
  unsigned int i;
  for (i=0;i<dim;i++)
  {  
    sum += (vec1[i]-vec2[i]) * (vec1[i]-vec2[i]);
  }  
  return sqrt(sum); 
}

float* gaussian_kernel_1d(float sigma, int dim)
{
  unsigned int i;  
  //printf("GaussianKernel1D(): Creating 1x%d vector for sigma=%.3f gaussian kernel/n", dim, sigma);  
   
  float *kern=(float*)malloc( dim*sizeof(float) );  
  float s2 = sigma * sigma;  
  int c = dim / 2;  
  float m= 1.0/(sqrt(2.0 * CV_PI) * sigma);  
  double v;   
  for ( i = 0; i < (dim + 1) / 2; i++)   
    {  
      v = m * exp(-(1.0*i*i)/(2.0 * s2)) ;  
      kern[c+i] = v;  
      kern[c-i] = v;  
    }  
  //   normalizeVec(kern, dim);  
  // for ( i = 0; i < dim; i++)  
  //  printf("%f  ", kern[i]);  
  //  printf("/n");  
  return kern;  
}

mat_c* gaussian_kernel_2d(float sigma)
{
  int dim;
  mat_c* mat;
  float s2;
  int i, j, c;
  float m, v;
  
  // int dim = (int) max(3.0f, GAUSSKERN * sigma);  
  dim = (int) fmaxf(3.0f, 2.0 * GAUSSKERN *sigma + 1.0f);  
  // make dim odd  
  if (dim % 2 == 0)
  {
    dim++;
  }  
  //printf("GaussianKernel(): Creating %dx%d matrix for sigma=%.3f gaussian/n", dim, dim, sigma);  
  mat = malloc_mat(dim, dim, 1, MAT_F2D);  
#define Mat(ROW,COL) (mat->data.f2d[(ROW)][(COL)])  
  s2 = sigma * sigma;  
  c = dim / 2;  
  //printf("%d %d/n", mat.size(), mat[0].size());  
  m= 1.0/(sqrt(2.0 * CV_PI) * sigma);  
  for (i = 0; i < (dim + 1) / 2; i++)   
    {  
      for (j = 0; j < (dim + 1) / 2; j++)   
	{  
	  //printf("%d %d %d/n", c, i, j);  
	  v = m * exp(-(1.0*i*i + 1.0*j*j) / (2.0 * s2));  
	  Mat(c+i,c+j) =v;  
	  Mat(c-i,c+j) =v;  
	  Mat(c+i,c-j) =v;  
	  Mat(c-i,c-j) =v;  
	}  
    }  
  // normalizeMat(mat);  
  return mat;  
}

float convolve_loc_width(float* kernel, int dim, mat_c* src, int x, int y)
{
#define Src(ROW,COL) (src->data.f2d[(ROW)][(COL)])  
  unsigned int i;  
  float pixel = 0;  
  int col;  
  int cen = dim / 2;  
  //printf("ConvolveLoc(): Applying convoluation at location (%d, %d)/n", x, y);  
  for ( i = 0; i < dim; i++)   
    {  
      col = x + (i - cen);  
      if (col < 0)  
	col = 0;  
      if (col >= src->cols)  
	col = src->cols - 1;  
      pixel += kernel[i] * Src(y,col);  
    }  
  if (pixel > 1)  
    pixel = 1;  
  return pixel;  
}

void convolve_1d_width(float* kern, int dim, mat_c* src, mat_c* dst)
{
#define DST(ROW,COL) (dst->data.f2d[(ROW)][(COL)])
  unsigned int i,j;  
   
  for ( j = 0; j < src->rows; j++)   
    {  
      for ( i = 0; i < src->cols; i++)   
	{  
	  //printf("%d, %d/n", i, j);  
	  DST(j,i) = convolve_loc_width(kern, dim, src, i, j);  
	}  
    }  
}

float convolve_loc_height(float* kernel, int dim, mat_c* src, int x, int y)
{
#define Src(ROW,COL) (src->data.f2d[(ROW)][(COL)])  
    unsigned int j;  
 float pixel = 0;  
 int cen = dim / 2;  
 //printf("ConvolveLoc(): Applying convoluation at location (%d, %d)/n", x, y);  
 for ( j = 0; j < dim; j++)   
 {  
  int row = y + (j - cen);  
  if (row < 0)  
   row = 0;  
  if (row >= src->rows)  
   row = src->rows - 1;  
  pixel += kernel[j] * Src(row,x);  
 }  
 if (pixel > 1)  
  pixel = 1;  
 return pixel;  
}

void convolve_1d_height(float* kern, int dim, mat_c* src, mat_c* dst)
{
#define DST(ROW,COL) (dst->data.f2d[(ROW)][(COL)])
    unsigned int i,j;  
 for ( j = 0; j < src->rows; j++)   
 {  
  for ( i = 0; i < src->cols; i++)   
  {  
   //printf("%d, %d/n", i, j);  
   DST(j,i) = convolve_loc_height(kern, dim, src, i, j);  
  }  
 }  
}

int blur_image(mat_c* src, mat_c* dst, float sigma) 
{
  float* convkernel;  
  int dim = (int) fmaxf(3.0f, 2.0 * GAUSSKERN * sigma + 1.0f);  
  mat_c *tempMat;  
  // make dim odd  
  if (dim % 2 == 0)
  { 
    dim++;
  }  
  tempMat = malloc_mat(src->rows, src->cols, 1, MAT_F2D);  
  convkernel = gaussian_kernel_1d(sigma, dim);  
  
   
  convolve_1d_width(convkernel, dim, src, tempMat);  
  convolve_1d_height(convkernel, dim, tempMat, dst);

  free_mat(tempMat);  
  return dim;  
}

void free_keypoint(keypoint *k)
{
  keypoint i,nt;
  i = *k;
  while(i)
  {
    nt = i->next;
    if(NULL != i->descrip)
    {
      free(i->descrip);
    }
    free(i);
    i = nt;
  }
  *k = NULL;
}

keypoint copy_keypoint(keypoint p)
{
  int i;
  keypoint n,rt;
  rt = (keypoint)malloc(sizeof(struct KeypointSt));
  n = rt;
  while(p)
  {
    n->row = p->row;
    n->col = p->col;
    n->sx = p->sx;
    n->sy = p->sy;
    n->octave = p->octave;
    n->level = p->level;
    n->scale = p->scale;
    n->ori = p->ori;
    n->mag = p->mag;
    n->descrip = (float*)malloc(LEN*sizeof(float));
    n->match = NULL;
    for(i=0;i<LEN;i++)
    {
      n->descrip[i] = p->descrip[i];
    }
    p = p->next;
    if(NULL != p)
    {
      n->next = (keypoint)malloc(sizeof(KeypointSt));    
      n = n->next;
    }
    else
    {
      n->next = NULL;
    }
    
  }
  return rt;
}
