#include "canny.h"

//Fraction of pixels that should be above the HIGH threshold
float ratio = 0.1f;
int WIDTH = 0;

inline int range(mat_c* x, int i, int j)
{
  if((i>=0) && (i<x->rows) && (j>0) && (j<x->cols))
  {
    return 1;
  }
  else
  {
    return 0;
  }
}

inline float norm(float x, float y)
{
  return (float) sqrt((double)(x*x + y*y));
}

void canny(float s, mat_c* im, mat_c* mag, mat_c* ori)
{
  int width;
  mat_c *smx, *smy;
  mat_c *dx, *dy;
  int i,j,n;
  float gau[MAX_MASK_SIZE], dgau[MAX_MASK_SIZE], z;
  
  //Create a Gaussian and a derivative of Gaussian filter mask
  for(i=0; i<MAX_MASK_SIZE; i++)
  {
    gau[i] = meanGauss((float)i, s);
    if(gau[i] < 0.005)
    {
      width = i;
      break;
    }
    dgau[i] = dGauss((float)i, s);
  }
  n = width+width+1;
  WIDTH = width/2;
  printf("Smoothing with a Gaussian (width = %d) ...\n", n);

  smx = malloc_mat(im->rows, im->cols, 1, MAT_F2D);
  smy = malloc_mat(im->rows, im->cols, 1, MAT_F2D);

  //Convolution of source image with a Gaussian in X and Y directions
  seperable_convolution(im, gau, width, smx, smy);
  
  // Now convolve smoothed data with a derivative
  printf("Convolution with the derivative of a Gaussian..\n");
  dx = malloc_mat(im->rows, im->cols, 1, MAT_F2D);
  
  dxy_seperable_convolution(smx, im->rows, im->cols, dgau, width, dx, 1);
  free_mat(smx);

  dy = malloc_mat(im->rows, im->cols, 1, MAT_F2D);
  dxy_seperable_convolution(smy, im->rows, im->cols, dgau, width, dy, 0);
  free_mat(smy);

  //create an image of the norm of dx,dy
  for(i=0; i<im->rows; i++)
  {
    for(j=0; j<im->cols; j++)
    {
      z = norm(dx->data.f2d[i][j], dy->data.f2d[i][j]);
      mag->data.ptr[i][j] = (unsigned char)(z*MAG_SCALE);
    }
  }
  //Non-maximum suppression edge pixels should be a local max
  nonmax_suppress(dx, dy, im->rows, im->cols, mag, ori);
  
  free_mat(dx);
  free_mat(dy);
}

//Gaussian
inline float gauss(float x, float sigma)
{
  float xx;
  
  if(0 == sigma)
  {
    return 0.0;
  }
  xx = (float)exp((double)((-x*x)/(2*sigma*sigma)));
  return xx;
}

inline float meanGauss(float x, float sigma)
{
  float z;
  z = (gauss(x,sigma) + gauss(x+0.5f,sigma) + gauss(x-0.5f,sigma))/3.0f;
  z = z/(PI*2.0f*sigma*sigma);
  return z;
}

//First derivative of Gaussian
inline float dGauss(float x, float sigma)
{
  return -x/(sigma*sigma) * gauss(x,sigma);
}

/*HYSTERESIS thersholding of edge pixels. 
 *Starting at pixels with a value greater than ther HIGH threshold,
 *trace a connected sequence of pixels that have a value greater than
 *the LOW threhsold.*/
void hysteresis(int high, int low, mat_c* im, mat_c* mag, mat_c* oriim)
{
  int i,j;
  printf("Beginning hysteresis thresholding...\n");
  for(i=0; i<im->rows; i++)
  {
    for(j=0;j<im->cols; j++)
    {
      im->data.ptr[i][j] = 0;
    }
  }
  
  if(high<low)
  {
    estimate_thresh(mag, &high, &low);
    printf("Hysteresis thresholds (from image): HI %d LOW %d\n", high, low);
  }

  //For each edge with a magnitude above the high threshold, 
  //begin tracing edge pixels that are above the low threshold.
  for(i=0; i<im->rows; i++)
  {
    for(j=0; j<im->cols; j++)
    {
      if(mag->data.ptr[i][j] >= high)
      {
	trace(i, j, low, im, mag, oriim);
      }
    }
  }

  //Make the edge 255 (to be the same as the other methods)
  for(i=0; i<im->rows; i++)
  {
    for(j=0; j<im->cols; j++)
    {
      if(0 == im->data.ptr[i][j])
      {
	im->data.ptr[i][j] = 0;
      }
      else
      {
	im->data.ptr[i][j] = 255;
      }
    }
  }

}

/* TRACE - recursively trace edge pixels that have a threshold > the low
 * edge threshold, continuing from the pixel at (i,j). */
int trace(int i, int j, int low, mat_c* im, mat_c* mag, mat_c* ori)
{
  int n,m;
  char flag = 0;
  
  if(0 == im->data.ptr[i][j])
  {
    im->data.ptr[i][j] = 255;
    flag = 0;
    for(n=-1;n<=1;n++)
    {
      for(m=-1;m<=1;m++)
      {
	if(0==i && 0==m)
	{
	  continue;
	}
	if(range(mag, i+n, j+m) && mag->data.ptr[i+n][j+m] >= low)
	{
	  if(trace(i+n, j+m, low, im, mag, ori))
	  {
	    flag = 1;
	    break;
	  }
	}
      }
      if(flag)
      {
	break;
      }
    }
    return 1;
  }
  return 0;
}

void seperable_convolution(mat_c* im, float * gau, int width, mat_c *smx, mat_c *smy)
{
  int i,j,k, I1, I2, nr, nc;
  float x, y;
  nr = im->rows;
  nc = im->cols;
  
  for(i=0; i<nr; i++)
  {
    for(j=0; j<nc; j++)
    {
      x = gau[0] * im->data.ptr[i][j];
      y = gau[0] * im->data.ptr[i][j];
      for(k=1; k<width; k++)
      {
	I1 = (i+k)%nr;
	I2 = (i-k+nr)%nr;
	y += gau[k]*im->data.ptr[I1][j] + gau[k]*im->data.ptr[I2][j];
	
	I1 = (j+k)%nc;
	I2 = (j-k+nc)%nc;
	x += gau[k]*im->data.ptr[i][I1] + gau[k]*im->data.ptr[i][I2];
      }
      smx->data.f2d[i][j] = x;
      smy->data.f2d[i][j] = y;
    }
  }

}

void dxy_seperable_convolution(mat_c* im, int nr, int nc, float *gau, int width, mat_c* sm, int which)
{
  int i,j,k, I1, I2;
  float x;

  for (i=0; i<nr; i++)
  {
    for (j=0; j<nc; j++)
    {
      x = 0.0;
      for (k=1; k<width; k++)
      {
	if (which == 0)
	{
	  I1 = (i+k)%nr; I2 = (i-k+nr)%nr;
	  x += -gau[k]*im->data.f2d[I1][j] + gau[k]*im->data.f2d[I2][j];
	}
	else
	{
	  I1 = (j+k)%nc; I2 = (j-k+nc)%nc;
	  x += -gau[k]*im->data.f2d[i][I1] + gau[k]*im->data.f2d[i][I2];
	}
      }
      sm->data.f2d[i][j] = x;
    }
  }
}

void nonmax_suppress(mat_c *dx, mat_c *dy, int nr, int nc, mat_c* mag, mat_c* ori)
{
  int i,j;
  float xx, yy, g2, g1, g3, g4, g, xc, yc;

  for (i=1; i<mag->rows-1; i++)
  {
    for (j=1; j<mag->cols-1; j++)
    {
      mag->data.ptr[i][j] = 0;
      
      /* Treat the x and y derivatives as components of a vector */
      xc = dx->data.f2d[i][j];
      yc = dy->data.f2d[i][j];
      if (fabs(xc)<0.01 && fabs(yc)<0.01)
      {
	continue;
      }

      g  = norm (xc, yc);
      
      /* Follow the gradient direction, as indicated by the direction of
	 the vector (xc, yc); retain pixels that are a local maximum. */

      if (fabs(yc) > fabs(xc))
      {

	/* The Y component is biggest, so gradient direction is basically UP/DOWN */
	xx = fabs(xc)/fabs(yc);
	yy = 1.0;

	g2 = norm (dx->data.f2d[i-1][j], dy->data.f2d[i-1][j]);
	g4 = norm (dx->data.f2d[i+1][j], dy->data.f2d[i+1][j]);
	if (xc*yc > 0.0)
	{
	  g3 = norm (dx->data.f2d[i+1][j+1], dy->data.f2d[i+1][j+1]);
	  g1 = norm (dx->data.f2d[i-1][j-1], dy->data.f2d[i-1][j-1]);
	}
	else
	{
	  g3 = norm (dx->data.f2d[i+1][j-1], dy->data.f2d[i+1][j-1]);
	  g1 = norm (dx->data.f2d[i-1][j+1], dy->data.f2d[i-1][j+1]);
	}
	
      }
      else
      {

	/* The X component is biggest, so gradient direction is basically LEFT/RIGHT */
	xx = fabs(yc)/fabs(xc);
	yy = 1.0;
	
	g2 = norm (dx->data.f2d[i][j+1], dy->data.f2d[i][j+1]);
	g4 = norm (dx->data.f2d[i][j-1], dy->data.f2d[i][j-1]);
	if (xc*yc > 0.0)
	{
	  g3 = norm (dx->data.f2d[i-1][j-1], dy->data.f2d[i-1][j-1]);
	  g1 = norm (dx->data.f2d[i+1][j+1], dy->data.f2d[i+1][j+1]);
	}
	else
	{
	  g1 = norm (dx->data.f2d[i-1][j+1], dy->data.f2d[i-1][j+1]);
	  g3 = norm (dx->data.f2d[i+1][j-1], dy->data.f2d[i+1][j-1]);
	}
      }

      /* Compute the interpolated value of the gradient magnitude */
      if ( (g > (xx*g1 + (yy-xx)*g2)) &&
	   (g > (xx*g3 + (yy-xx)*g4)) )
      {
	if (g*MAG_SCALE <= 255)
	{
	  mag->data.ptr[i][j] = (unsigned char)(g*MAG_SCALE);
	}
	else
	{
	  mag->data.ptr[i][j] = 255;
	  ori->data.ptr[i][j] = (unsigned char) (atan2 (yc, xc) * ORI_SCALE);
	}
      }
      else
      {
	mag->data.ptr[i][j] = 0;
	ori->data.ptr[i][j] = 0;
      }

    }
  }
}

void estimate_thresh(mat_c* mag, int *hi, int *low)
{
  int i,j,k, hist[256], count;

  /* Build a histogram of the magnitude image. */
  for (k=0; k<256; k++)
  {
    hist[k] = 0;
  }

  for (i=WIDTH; i<mag->rows-WIDTH; i++)
  {
    for (j=WIDTH; j<mag->cols-WIDTH; j++)
    {
      hist[mag->data.ptr[i][j]]++;
    }
  }

  /* The high threshold should be > 80 or 90% of the pixels 
     j = (int)(ratio*mag->info->nr*mag->info->nc);
  */
  j = mag->rows;
  if (j<mag->cols)
  {
    j = mag->cols;
  }
  j = (int)(0.9*j);
  k = 255;

  count = hist[255];
  while (count < j)
  {
    k--;
    if (k<0)
    {
      break;
    }
    count += hist[k];
  }
  *hi = k;

  i=0;
  while (hist[i]==0)
  {
    i++;
  }

  *low = (*hi+i)/2.0f;
}

int canny_edge(mat_c* im)
{
  int i,j;
  float s = SIGMA;
  int low = LOW_THOD;
  int high = HIGH_THOD;
  mat_c *magim, *oriim;
  

  //Create local image space
  magim = malloc_mat(im->rows, im->cols, 1, MAT_UC2D);

  oriim = malloc_mat(im->rows, im->cols, 1, MAT_UC2D);
  
  //Apply the filter
  canny(s, im, magim, oriim);

  //Hysteresis thresholding of edge pixels
  hysteresis(high, low, im, magim, oriim);
  
  for(i=0; i<WIDTH; i++)
  {
    for(j=0;j<im->cols;j++)
    {
      im->data.ptr[i][j] = 0;
    }
  }

  for(i=im->rows-1; i>im->rows-1-WIDTH; i--)
  {
    for(j=0;j<im->cols;j++)
    {
      im->data.ptr[i][j] = 0;
    }
  }

  for(i=0; i<im->rows; i++)
  {
    for(j=0;j<WIDTH;j++)
    {
      im->data.ptr[i][j] = 0;
    }
  }

  for(i=0; i<im->rows; i++)
  {
    for(j=im->cols-WIDTH-1;j<im->cols;j++)
    {
      im->data.ptr[i][j] = 0;
    }
  }

  free_mat(magim);
  free_mat(oriim);
  
  return 1;
}
