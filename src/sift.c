#include "sift.h"

int numoctaves;  

image_octaves *DOGoctaves;        
//DOG pyr，DOG approx of LoG。  
    
//image_octaves *mag_thresh;  
image_octaves *mag_pyr;  
image_octaves *grad_pyr;  
    
//Keypoints  
keypoint keypoints=NULL;      //temp 
keypoint keyDescriptors=NULL; //final

void free_image_octaves(image_octaves *p, int numo, int numl)
{
  int i, j;
  for(i=0;i<numo;i++)
  {
    for(j=0;j<numl;j++)
    {
      free_mat(p[i].octave[j].level);
    }
    free(p[i].octave);
  }
  free(p);
  p = NULL;
}

mat_c *scale_init_image(mat_c* im)
{
  double sigma,preblur_sigma;  
  mat_c *imMat;  
  mat_c * dst;  
  mat_c *tempMat;  
  //filtering  
  imMat = malloc_mat(im->rows, im->cols, 1, MAT_F2D);  
  blur_image(im, imMat, INITSIGMA);
  
 
  //two opts: double the size or base on the ori size  
  //build the bottom
  if (DOUBLE_BASE_IMAGE_SIZE)   
  {  
    tempMat = doubleSizeImage2(imMat); 
#define TEMPMAT(ROW,COL) (tempMat->data.f2d[(ROW)][(COL)])  
    
    dst = malloc_mat(tempMat->rows, tempMat->cols, 1, MAT_F2D);  
    preblur_sigma = 1.0;//sqrt(2 - 4*INITSIGMA*INITSIGMA);  
    blur_image(tempMat, dst, preblur_sigma);   
    
    // The initial blurring for the first image of the first octave of the pyramid.  
    sigma = sqrt( (4*INITSIGMA*INITSIGMA) + preblur_sigma * preblur_sigma );  
    //  sigma = sqrt(SIGMA * SIGMA - INITSIGMA * INITSIGMA * 4);  
    //printf("Init Sigma: %f/n", sigma);  
    blur_image(dst, tempMat, sigma);
    free_mat(dst);   
    return tempMat;  
  }   
  else   
  {  
    dst = malloc_mat(im->rows, im->cols, 1, MAT_F2D);  
    //sigma = sqrt(SIGMA * SIGMA - INITSIGMA * INITSIGMA);  
    preblur_sigma = 1.0;//sqrt(2 - 4*INITSIGMA*INITSIGMA);  
    sigma = sqrt( (4*INITSIGMA*INITSIGMA) + preblur_sigma * preblur_sigma );  
    //printf("Init Sigma: %f/n", sigma);  
    blur_image(imMat, dst, sigma);
    return dst;  
  }   
}

image_octaves* build_gaussian_octaves(mat_c* image)
{
  image_octaves *octaves;  
  mat_c *tempMat;  
  mat_c *dst;  
  mat_c *temp;  
   
  int i,j;  
  double k = pow(2, 1.0/((float)SCALESPEROCTAVE));   
  float preblur_sigma, initial_sigma , sigma1,sigma2,sigma,absolute_sigma,sigma_f;  
  
  int dim = fminf(image->rows, image->cols);  
  //int numoctaves = (int) (log((double) dim) / log(2.0)) - 2;    //number of octives  
  
  //numoctaves = fminf(numoctaves, MAXOCTAVES);
  numoctaves = MAXOCTAVES;
   
  octaves=(image_octaves*) malloc( numoctaves * sizeof(image_octaves) );  
  DOGoctaves=(image_octaves*) malloc( numoctaves * sizeof(image_octaves) );  
   
  //printf("BuildGaussianOctaves(): Base image dimension is %dx%d\n", (int)(0.5*(image->cols)), (int)(0.5*(image->rows)) );  
  //printf("BuildGaussianOctaves(): Building %d octaves\n", numoctaves);  
   
  // start with initial source image  
  tempMat = copy_mat( image );
  // preblur_sigma = 1.0;//sqrt(2 - 4*INITSIGMA*INITSIGMA);  
  initial_sigma = sqrt(2);//sqrt( (4*INITSIGMA*INITSIGMA) + preblur_sigma * preblur_sigma );  
  //   initial_sigma = sqrt(SIGMA * SIGMA - INITSIGMA * INITSIGMA * 4);  
      
  //different levels in a octave 
  for ( i = 0; i < numoctaves; i++)   
  {       
    //printf("Building octave %d of dimesion (%d, %d)\n", i, tempMat->cols,tempMat->rows);  

    octaves[i].octave= (image_levels*) malloc( (SCALESPEROCTAVE + 3) * sizeof(image_levels) );  
    DOGoctaves[i].octave= (image_levels*) malloc( (SCALESPEROCTAVE + 2) * sizeof(image_levels) );  
 
      (octaves[i].octave)[0].level=tempMat;  
    
      octaves[i].col=tempMat->cols;  
      octaves[i].row=tempMat->rows;  
      DOGoctaves[i].col=tempMat->cols;  
      DOGoctaves[i].row=tempMat->rows;  
      if (DOUBLE_BASE_IMAGE_SIZE)  
	octaves[i].subsample=pow(2,i)*0.5;  
      else  
	octaves[i].subsample=pow(2,i);  
    
      if(i==0)       
	{  
	  (octaves[0].octave)[0].levelsigma = initial_sigma;  
	  (octaves[0].octave)[0].absolute_sigma = initial_sigma;  
	  //printf("0 scale and blur sigma : %f \n", (octaves[0].subsample) * ((octaves[0].octave)[0].absolute_sigma));  
	}  
      else  
	{  
	  (octaves[i].octave)[0].levelsigma = (octaves[i-1].octave)[SCALESPEROCTAVE].levelsigma;  
	  (octaves[i].octave)[0].absolute_sigma = (octaves[i-1].octave)[SCALESPEROCTAVE].absolute_sigma;  
	  //printf( "0 scale and blur sigma : %f \n", ((octaves[i].octave)[0].absolute_sigma) );  
	}  
      sigma = initial_sigma;  
 
      for ( j =  1; j < SCALESPEROCTAVE + 3; j++)   
	{  
	  dst = malloc_mat(tempMat->rows, tempMat->cols, 1, MAT_F2D); 
	  temp = malloc_mat(tempMat->rows, tempMat->cols, 1, MAT_F2D);//DOG  
	  // 2 passes of 1D on original  
	  //   if(i!=0)  
	  //   {  
	  //       sigma1 = pow(k, j - 1) * ((octaves[i-1].octave)[j-1].levelsigma);  
	  //          sigma2 = pow(k, j) * ((octaves[i].octave)[j-1].levelsigma);  
	  //       sigma = sqrt(sigma2*sigma2 - sigma1*sigma1);  
	  sigma_f= sqrt(k*k-1)*sigma;  
	  //   }  
	  //   else  
	  //   {  
	  //       sigma = sqrt(SIGMA * SIGMA - INITSIGMA * INITSIGMA * 4)*pow(k,j);  
	  //   }    
	  sigma = k*sigma;  
	  absolute_sigma = sigma * (octaves[i].subsample);  
	  //printf("%d scale and Blur sigma: %f  \n", j, absolute_sigma);  
     
	  (octaves[i].octave)[j].levelsigma = sigma;  
	  (octaves[i].octave)[j].absolute_sigma = absolute_sigma;  
	  //gaus  
	  int length=blur_image((octaves[i].octave)[j-1].level, dst, sigma_f);
	  (octaves[i].octave)[j].levelsigmalength = length;  
	  (octaves[i].octave)[j].level=dst;  
	  //DOG  
	  sub_mat( ((octaves[i].octave)[j]).level, ((octaves[i].octave)[j-1]).level, temp);  
	  //         cvAbsDiff( ((octaves[i].Octave)[j]).Level, ((octaves[i].Octave)[j-1]).Level, temp );  
	  ((DOGoctaves[i].octave)[j-1]).level=temp;  
	}  
      // halve the image size for next iteration  
      tempMat  = half_size_image( ( (octaves[i].octave)[SCALESPEROCTAVE].level ) );
    }
  return octaves;  
}

int detect_keypoint(int numoctaves, image_octaves *gaussianPyr)
{
  double curvature_threshold;  
  curvature_threshold= ((CURVATURE_THRESHOLD + 1)*(CURVATURE_THRESHOLD + 1))/CURVATURE_THRESHOLD;  
#define ImLevelsD(OCTAVE,LEVEL,ROW,COL) (DOGoctaves[(OCTAVE)].octave[(LEVEL)].level->data.f2d[(ROW)][(COL)])  
   
  int   keypoint_count = 0;   
  keypoints = NULL;
  keyDescriptors=NULL; 
  
  for (int i=0; i<numoctaves; i++)    
  {          
    for(int j=1;j<SCALESPEROCTAVE+1;j++)//scaleperoctave in middle 
    {    
      //float sigma=(GaussianPyr[i].Octave)[j].levelsigma;  
      //int dim = (int) (max(3.0f, 2.0*GAUSSKERN *sigma + 1.0f)*0.5);  
      int dim = (int)(0.5*((gaussianPyr[i].octave)[j].levelsigmalength)+0.5);  
      for (int m=dim;m<((DOGoctaves[i].row)-dim);m++)
      {   
	for(int n=dim;n<((DOGoctaves[i].col)-dim);n++)  
	{       
	  if ( fabs(ImLevelsD(i,j,m,n))>= CONTRAST_THRESHOLD )  
	  {  
	    
	    if ( ImLevelsD(i,j,m,n)!=0.0 )  //1
	    {  
	      float inf_val=ImLevelsD(i,j,m,n);  
	      if(( (inf_val <= ImLevelsD(i,j-1,m-1,n-1))&&  
		   (inf_val <= ImLevelsD(i,j-1,m  ,n-1))&&  
		   (inf_val <= ImLevelsD(i,j-1,m+1,n-1))&&  
		   (inf_val <= ImLevelsD(i,j-1,m-1,n  ))&&  
		   (inf_val <= ImLevelsD(i,j-1,m  ,n  ))&&  
		   (inf_val <= ImLevelsD(i,j-1,m+1,n  ))&&  
		   (inf_val <= ImLevelsD(i,j-1,m-1,n+1))&&  
		   (inf_val <= ImLevelsD(i,j-1,m  ,n+1))&&  
		   (inf_val <= ImLevelsD(i,j-1,m+1,n+1))&&    //9  
		   
		   (inf_val <= ImLevelsD(i,j,m-1,n-1))&&  
		   (inf_val <= ImLevelsD(i,j,m  ,n-1))&&  
		   (inf_val <= ImLevelsD(i,j,m+1,n-1))&&  
		   (inf_val <= ImLevelsD(i,j,m-1,n  ))&&  
		   (inf_val <= ImLevelsD(i,j,m+1,n  ))&&  
		   (inf_val <= ImLevelsD(i,j,m-1,n+1))&&  
		   (inf_val <= ImLevelsD(i,j,m  ,n+1))&&  
		   (inf_val <= ImLevelsD(i,j,m+1,n+1))&&     //8  
		   
		   (inf_val <= ImLevelsD(i,j+1,m-1,n-1))&&  
		   (inf_val <= ImLevelsD(i,j+1,m  ,n-1))&&  
		   (inf_val <= ImLevelsD(i,j+1,m+1,n-1))&&  
		   (inf_val <= ImLevelsD(i,j+1,m-1,n  ))&&  
		   (inf_val <= ImLevelsD(i,j+1,m  ,n  ))&&  
		   (inf_val <= ImLevelsD(i,j+1,m+1,n  ))&&  
		   (inf_val <= ImLevelsD(i,j+1,m-1,n+1))&&  
		   (inf_val <= ImLevelsD(i,j+1,m  ,n+1))&&  
		   (inf_val <= ImLevelsD(i,j+1,m+1,n+1))     //next 9          
		   ) ||   
		 ( (inf_val >= ImLevelsD(i,j-1,m-1,n-1))&&  
		   (inf_val >= ImLevelsD(i,j-1,m  ,n-1))&&  
		   (inf_val >= ImLevelsD(i,j-1,m+1,n-1))&&  
		   (inf_val >= ImLevelsD(i,j-1,m-1,n  ))&&  
		   (inf_val >= ImLevelsD(i,j-1,m  ,n  ))&&  
		   (inf_val >= ImLevelsD(i,j-1,m+1,n  ))&&  
		   (inf_val >= ImLevelsD(i,j-1,m-1,n+1))&&  
		   (inf_val >= ImLevelsD(i,j-1,m  ,n+1))&&  
		   (inf_val >= ImLevelsD(i,j-1,m+1,n+1))&&  
		   
		   (inf_val >= ImLevelsD(i,j,m-1,n-1))&&  
		   (inf_val >= ImLevelsD(i,j,m  ,n-1))&&  
		   (inf_val >= ImLevelsD(i,j,m+1,n-1))&&  
		   (inf_val >= ImLevelsD(i,j,m-1,n  ))&&  
		   (inf_val >= ImLevelsD(i,j,m+1,n  ))&&  
		   (inf_val >= ImLevelsD(i,j,m-1,n+1))&&  
		   (inf_val >= ImLevelsD(i,j,m  ,n+1))&&  
		   (inf_val >= ImLevelsD(i,j,m+1,n+1))&&   
		   
		   (inf_val >= ImLevelsD(i,j+1,m-1,n-1))&&  
		   (inf_val >= ImLevelsD(i,j+1,m  ,n-1))&&  
		   (inf_val >= ImLevelsD(i,j+1,m+1,n-1))&&  
		   (inf_val >= ImLevelsD(i,j+1,m-1,n  ))&&  
		   (inf_val >= ImLevelsD(i,j+1,m  ,n  ))&&  
		   (inf_val >= ImLevelsD(i,j+1,m+1,n  ))&&  
		   (inf_val >= ImLevelsD(i,j+1,m-1,n+1))&&  
		   (inf_val >= ImLevelsD(i,j+1,m  ,n+1))&&  
		   (inf_val >= ImLevelsD(i,j+1,m+1,n+1))   
		   ) )      //2、fit 26  
	      {     
		//CONTRAST_THRESHOLD=0.02  
		if ( fabs(ImLevelsD(i,j,m,n))>= CONTRAST_THRESHOLD )  
		{  
		  //CURVATURE_THRESHOLD=10.0，Hessian  
		  // Compute the entries of the Hessian matrix at the extrema location.  
		  /* 
		     1   0   -1 
		     0   0   0 
		     -1   0   1         *0.25 
		  */  
		  // Compute the trace and the determinant of the Hessian.  
		  //Tr_H = Dxx + Dyy;  
		  //Det_H = Dxx*Dyy - Dxy^2;  
		  float Dxx,Dyy,Dxy,Tr_H,Det_H,curvature_ratio;  
		  Dxx = ImLevelsD(i,j,m,n-1) + ImLevelsD(i,j,m,n+1)-2.0*ImLevelsD(i,j,m,n);  
		  Dyy = ImLevelsD(i,j,m-1,n) + ImLevelsD(i,j,m+1,n)-2.0*ImLevelsD(i,j,m,n);  
		  Dxy = ImLevelsD(i,j,m-1,n-1) + ImLevelsD(i,j,m+1,n+1) - ImLevelsD(i,j,m+1,n-1) - ImLevelsD(i,j,m-1,n+1);  
		  Tr_H = Dxx + Dyy;  
		  Det_H = Dxx*Dyy - Dxy*Dxy;  
		  // Compute the ratio of the principal curvatures.  
		  curvature_ratio = (1.0*Tr_H*Tr_H)/Det_H;  
		  if ( (Det_H>=0.0) && (curvature_ratio <= curvature_threshold) ) 
		  {  
	
		    keypoint_count++;  
		    keypoint k;  
		    /* Allocate memory for the keypoint. */  
		    k = (keypoint) malloc(sizeof(struct KeypointSt));  
		    k->next = keypoints;  
		    keypoints = k;  
		    k->row = m*(gaussianPyr[i].subsample);  
		    k->col =n*(gaussianPyr[i].subsample);  
		    k->sy = m;    //rows  
		    k->sx = n;    //cols  
		    k->octave=i;  
		    k->level=j;  
		    k->scale = (gaussianPyr[i].octave)[j].absolute_sigma;
		    k->descrip = NULL;
		  }//if >curvature_thresh  
		}//if >contrast  
	      }//if inf value  
	    }//if non zero  
	  }//if >contrast  
	}  //for concrete image level col 
      }
    }//for levels  
  }//for octaves  
  return keypoint_count;
}

void compute_grad_direcand_mag(int numoctaves, image_octaves *gaussianPyr)
{
  // ImageOctaves *mag_thresh ;  
  mag_pyr=(image_octaves*) malloc( numoctaves * sizeof(image_octaves) );  
  grad_pyr=(image_octaves*) malloc( numoctaves * sizeof(image_octaves) );  
  // float sigma=( (GaussianPyr[0].Octave)[SCALESPEROCTAVE+2].absolute_sigma ) / GaussianPyr[0].subsample;  
  // int dim = (int) (max(3.0f, 2 * GAUSSKERN *sigma + 1.0f)*0.5+0.5);  
#define ImLevels(OCTAVE,LEVEL,ROW,COL) (gaussianPyr[(OCTAVE)].octave[(LEVEL)].level->data.f2d[(ROW)][(COL)])  
  for (int i=0; i<numoctaves; i++)    
  {          
    mag_pyr[i].octave= (image_levels*) malloc( (SCALESPEROCTAVE) * sizeof(image_levels) );  
    grad_pyr[i].octave= (image_levels*) malloc( (SCALESPEROCTAVE) * sizeof(image_levels) );  
    for(int j=1;j<SCALESPEROCTAVE+1;j++)//scaleperoctave  
    {    
      mat_c *Mag = malloc_mat(gaussianPyr[i].row, gaussianPyr[i].col, 1, MAT_F2D);  
      mat_c *Ori = malloc_mat(gaussianPyr[i].row, gaussianPyr[i].col, 1, MAT_F2D);  
      mat_c *tempMat1 = malloc_mat(gaussianPyr[i].row, gaussianPyr[i].col, 1, MAT_F2D);  
      mat_c *tempMat2 = malloc_mat(gaussianPyr[i].row, gaussianPyr[i].col, 1, MAT_F2D);  
      zero_mat(Mag);  
      zero_mat(Ori);  
      zero_mat(tempMat1);  
      zero_mat(tempMat2);   
#define MAG(ROW,COL) (Mag->data.f2d[(ROW)][(COL)])     
#define ORI(ROW,COL) (Ori->data.f2d[(ROW)][(COL)])    
#define TEMPMAT1(ROW,COL) (tempMat1->data.f2d[(ROW)][(COL)])  
#define TEMPMAT2(ROW,COL) (tempMat2->data.f2d[(ROW)][(COL)])  
      for (int m=1;m<(gaussianPyr[i].row-1);m++) 
      {  
	for(int n=1;n<(gaussianPyr[i].col-1);n++)  
	{  
		 
	  TEMPMAT1(m,n) = 0.5*( ImLevels(i,j,m,n+1)-ImLevels(i,j,m,n-1) );  //dx  
	  TEMPMAT2(m,n) = 0.5*( ImLevels(i,j,m+1,n)-ImLevels(i,j,m-1,n) );  //dy  
	  MAG(m,n) = sqrt(TEMPMAT1(m,n)*TEMPMAT1(m,n)+TEMPMAT2(m,n)*TEMPMAT2(m,n));  //mag  
	  
	  ORI(m,n) =atan( TEMPMAT2(m,n)/TEMPMAT1(m,n) );  
	  if (ORI(m,n)==PI)
	  {  
	    ORI(m,n)=-PI;  
	  }  
	}
      }
      ((mag_pyr[i].octave)[j-1]).level=Mag;  
      ((grad_pyr[i].octave)[j-1]).level=Ori;  
      free_mat(tempMat1);  
      free_mat(tempMat2);  
    }//for levels 
  }//for octaves
}

void assign_the_main_orientation(int numoctaves, image_octaves *gaussianPyr,image_octaves *mag_pyr,image_octaves *grad_pyr)
{
  // Set up the histogram bin centers for a 36 bin histogram.  
  int num_bins = 36;  
  float hist_step = 2.0*PI/num_bins;  
  float hist_orient[36];
  float sigma1;
  int zero_pad;
  int keypoint_count = 0;
  keypoint p = keypoints;
  
  for (int i=0;i<36;i++)  
  {
    hist_orient[i]=-PI+i*hist_step;
  }  
  sigma1=( ((gaussianPyr[0].octave)[SCALESPEROCTAVE].absolute_sigma) ) / (gaussianPyr[0].subsample);//SCALESPEROCTAVE+2  
  zero_pad = (int) (fmaxf(3.0f, 2 * GAUSSKERN *sigma1 + 1.0f)*0.5+0.5);  
  //Assign orientations to the keypoints.  
#define ImLevels2(OCTAVES,LEVELS,ROW,COL) (gaussianPyr[(OCTAVES)].octave[(LEVELS)].level->data.f2d[(ROW)][(COL)])  
  while(p) // not the end
  {  
    int i=p->octave;  
    int j=p->level;  
    int m=p->sy;   //rows  
    int n=p->sx;   //cols
    if ((m>=zero_pad)&&(m<gaussianPyr[i].row-zero_pad)&&  
	(n>=zero_pad)&&(n<gaussianPyr[i].col-zero_pad) )  
    {  
      float sigma=( ((gaussianPyr[i].octave)[j].absolute_sigma) ) / (gaussianPyr[i].subsample);  
        
      mat_c* mat = gaussian_kernel_2d( sigma );           
      int dim=(int)(0.5 * (mat->rows));

      double maxGrad = 0.0;  
      int maxBin = 0;
      int b; 
     
      double* orienthist;
      
      bool binIsKeypoint[36]; 
      double oneBinRad = (2.0 * PI) / 36;

      double maxPeakValue=0.0;  
      double maxDegreeCorrection=0.0; 
      
#define MAT(ROW,COL) (mat->data.f2d[(ROW)][(COL)]) 
    
      orienthist = (double*)malloc(36 * sizeof(double));
      for ( int sw = 0 ; sw < 36 ; ++sw)   
      {  
	orienthist[sw]=0.0;    
      }  
      
      for (int x=m-dim,mm=0;x<(m+dim);x++,mm++)
      {   
	for(int y=n-dim,nn=0;y<(n+dim);y++,nn++)  
	  {       
	    double dx = 0.5*(ImLevels2(i,j,x,y+1)-ImLevels2(i,j,x,y-1));  //dx
	    double dy = 0.5*(ImLevels2(i,j,x+1,y)-ImLevels2(i,j,x-1,y));  //dy
	    double mag = sqrt(dx*dx+dy*dy);  //mag  
	    double Ori =atan( 1.0*dy/dx );  
	    int binIdx = find_closest_rotation_bin(36, Ori);
	    
	    orienthist[binIdx] = orienthist[binIdx] + 1.0* mag * MAT(mm,nn);
	  } 
      } 
      
      // Find peaks in the orientation histogram using nonmax suppression.  
      average_weak_bins (orienthist, 36);
      
      // find the maximum peak in gradient orientation  
      //double maxGrad = 0.0;  
      //int maxBin = 0;
      //int b
      for (b = 0 ; b < 36 ; ++b)   
      {  
	if (orienthist[b] > maxGrad)   
	{  
	  maxGrad = orienthist[b];  
	  maxBin = b;  
	}  
      }  
      // First determine the real interpolated peak high at the maximum bin  
      // position, which is guaranteed to be an absolute peak.  
      
      if ( (interpolate_orientation ( orienthist[maxBin == 0 ? (36 - 1) : (maxBin - 1)],  
				     orienthist[maxBin], orienthist[(maxBin + 1) % 36],  
				     &maxDegreeCorrection, &maxPeakValue)) == false)
      {  
	printf("BUG: Parabola fitting broken\n");  
      }
      
      // Now that we know the maximum peak value, we can find other keypoint  
      // orientations, which have to fulfill two criterias:  
      //  
      //  1. They must be a local peak themselves. Else we might add a very  
      //     similar keypoint orientation twice (imagine for example the  
      //     values: 0.4 1.0 0.8, if 1.0 is maximum peak, 0.8 is still added  
      //     with the default threshhold, but the maximum peak orientation  
      //     was already added).  
      //  2. They must have at least peakRelThresh times the maximum peak  
      //     value.  
      //bool binIsKeypoint[36];  
      for ( b = 0 ; b < 36 ; ++b)   
      {  
	int leftI;
	int rightI;
	
	binIsKeypoint[b] = false;  
	// The maximum peak of course is  
	if (b == maxBin)   
	{  
	  binIsKeypoint[b] = true;  
	  continue;  
	}  
	// Local peaks are, too, in case they fulfill the threshhold  
	if (orienthist[b] < (PEAK_REL_THRESH * maxPeakValue))  
	  continue;  
	leftI = (b == 0) ? (36 - 1) : (b - 1);  
	rightI = (b + 1) % 36;  
	if (orienthist[b] <= orienthist[leftI] || orienthist[b] <= orienthist[rightI])  
	  continue; // no local peak  
	binIsKeypoint[b] = true;  
      }  
      // find other possible locations  
      //double oneBinRad = (2.0 * PI) / 36;  
      for ( b = 0 ; b < 36 ; ++b)   
      {  
	if (binIsKeypoint[b] == false)
	{
	  continue;  
	}
	
	int bLeft = (b == 0) ? (36 - 1) : (b - 1);  
	int bRight = (b + 1) % 36;  
	// Get an interpolated peak direction and value guess.  
	double peakValue;  
	double degreeCorrection;  
	
	double maxPeakValue, maxDegreeCorrection;                
	if (interpolate_orientation ( orienthist[maxBin == 0 ? (36 - 1) : (maxBin - 1)],  
				     orienthist[maxBin], orienthist[(maxBin + 1) % 36],  
				     &degreeCorrection, &peakValue) == false)  
	{  
	  printf("BUG: Parabola fitting broken\n");  
	}  
       
	double degree = (b + degreeCorrection) * oneBinRad - PI;  
	if (degree < -PI)
	{  
	  degree += 2.0 * PI;
	}  
	else if (degree > PI)
	{  
	  degree -= 2.0 * PI;
	}  

	keypoint k;
	/* Allocate memory for the keypoint Descriptor. */  
	k = (keypoint) malloc(sizeof(struct KeypointSt)); 
	k->next = keyDescriptors;  
	keyDescriptors = k;  
	k->descrip = (float*)malloc(LEN * sizeof(float));  
	k->row = p->row;  
	k->col = p->col;  
	k->sy = p->sy;    //rows  
	k->sx = p->sx;    //cols  
	k->octave = p->octave;  
	k->level = p->level;  
	k->scale = p->scale;        
	k->ori = degree;  
	k->mag = peakValue;    
	
      }//for  
      free(orienthist);  
    }
    p=p->next;
  } 
}

int find_closest_rotation_bin (int binCount, float angle)
{
  angle += CV_PI;  
  angle /= 2.0 * CV_PI;  
  // calculate the aligned bin  
  angle *= binCount;  
  int idx = (int) angle;  
  if (idx == binCount)
  {  
    idx = 0;
  }  
  return (idx);  
}

// Average the content of the direction bins. 
void average_weak_bins (double* hist, int binCount)
{
  // TODO: make some tests what number of passes is the best. (its clear  
  // one is not enough, as we may have something like  
  // ( 0.4, 0.4, 0.3, 0.4, 0.4 ))  
  for (int sn = 0 ; sn < 2 ; ++sn)   
  {  
    double firstE = hist[0];  
    double last = hist[binCount-1];  
    for (int sw = 0 ; sw < binCount ; ++sw)   
    {  
      double cur = hist[sw];  
      double next = (sw == (binCount - 1)) ? firstE : hist[(sw + 1) % binCount];  
      hist[sw] = (last + cur + next) / 3.0;  
      last = cur;  
    }  
  }  
}

// Fit a parabol to the three points (-1.0 ; left), (0.0 ; middle) and  
// (1.0 ; right).  
// Formulas:  
// f(x) = a (x - c)^2 + b  
// c is the peak offset (where f'(x) is zero), b is the peak value.  
// In case there is an error false is returned, otherwise a correction  
// value between [-1 ; 1] is returned in 'degreeCorrection', where -1  
// means the peak is located completely at the left vector, and -0.5 just  
// in the middle between left and middle and > 0 to the right side. In  
// 'peakValue' the maximum estimated peak value is stored.  
bool interpolate_orientation (double left, double middle,double right, double *degreeCorrection, double *peakValue)
{
  double a = ((left + right) - 2.0 * middle) / 2.0;   
  // degreeCorrection = peakValue = Double.NaN;  
   
  // Not a parabol  
  if (a == 0.0)  
    return false;  
  double c = (((left - middle) / a) - 1.0) / 2.0;  
  double b = middle - c * c * a;  
  if (c < -0.5 || c > 0.5)  
    return false;  
  *degreeCorrection = c;  
  *peakValue = b;  
  return true;  
}

void extract_feature_descriptors(int numoctaves, image_octaves *gaussianPyr)
{
  // The orientation histograms have 8 bins  
  int i,j,m;
  float orient_bin_spacing = PI/4;  
  float orient_angles[8]={-PI,-PI+orient_bin_spacing,-PI*0.5, -orient_bin_spacing,  
			  0.0, orient_bin_spacing, PI*0.5,  PI+orient_bin_spacing};  

  float *feat_grid=(float *) malloc( 2*16 * sizeof(float));  
  for (i=0;i<GridSpacing;i++)  
  {  
    for (j=0;j<2*GridSpacing;++j,++j)  
    {  
      feat_grid[i*2*GridSpacing+j]=-6.0+i*GridSpacing;  
      feat_grid[i*2*GridSpacing+j+1]=-6.0+0.5*j*GridSpacing;  
    }  
  }  
  
  float *feat_samples=(float *) malloc( 2*256 * sizeof(float));  
  for ( i=0;i<4*GridSpacing;i++)  
  {  
    for (j=0;j<8*GridSpacing;j+=2)  
    {  
      feat_samples[i*8*GridSpacing+j]=-(2*GridSpacing-0.5)+i;  
      feat_samples[i*8*GridSpacing+j+1]=-(2*GridSpacing-0.5)+0.5*j;  
    }  
  }  
  float feat_window = 2*GridSpacing;  
  keypoint p = keyDescriptors; 
  while(p) 
  {  
    float scale=(gaussianPyr[p->octave].octave)[p->level].absolute_sigma;  
    
    float sine = sin(p->ori);  
    float cosine = cos(p->ori);    
   
    float *featcenter=(float *) malloc( 2*16 * sizeof(float));  
    for (i=0;i<GridSpacing;i++)  
    {  
      for (j=0;j<2*GridSpacing;j+=2)  
      {  
	float x=feat_grid[i*2*GridSpacing+j];  
	float y=feat_grid[i*2*GridSpacing+j+1];  
	featcenter[i*2*GridSpacing+j]=((cosine * x + sine * y) + p->sx);  
	featcenter[i*2*GridSpacing+j+1]=((-sine * x + cosine * y) + p->sy);  
      }  
    }  
    // calculate sample window coordinates (rotated along keypoint)  
    float *feat=(float *) malloc( 2*256 * sizeof(float));  
    for ( i=0;i<64*GridSpacing;i++,i++)  
    {  
      float x=feat_samples[i];  
      float y=feat_samples[i+1];  
      feat[i]=((cosine * x + sine * y) + p->sx);  
      feat[i+1]=((-sine * x + cosine * y) + p->sy);  
    }  
    //Initialize the feature descriptor.  
    float *feat_desc = (float *) malloc( 128 * sizeof(float));  
    for (i=0;i<128;i++)  
    {  
      feat_desc[i]=0.0;  
      // printf("%f  ",feat_desc[i]);    
    }  
    //printf("/n");  
    for ( i=0;i<512;++i,++i)  
    {  
      float x_sample = feat[i];  
      float y_sample = feat[i+1];  
      // Interpolate the gradient at the sample position  
      /* 
	 0   1   0 
	 1   *   1 
	 0   1   0  
      */  
      float sample12=get_pixel_bi(((gaussianPyr[p->octave].octave)[p->level]).level, x_sample, y_sample-1);  
      float sample21=get_pixel_bi(((gaussianPyr[p->octave].octave)[p->level]).level, x_sample-1, y_sample);   
      float sample22=get_pixel_bi(((gaussianPyr[p->octave].octave)[p->level]).level, x_sample, y_sample);   
      float sample23=get_pixel_bi(((gaussianPyr[p->octave].octave)[p->level]).level, x_sample+1, y_sample);   
      float sample32=get_pixel_bi(((gaussianPyr[p->octave].octave)[p->level]).level, x_sample, y_sample+1);   
      //float diff_x = 0.5*(sample23 - sample21);  
      //float diff_y = 0.5*(sample32 - sample12);  
      float diff_x = sample23 - sample21;  
      float diff_y = sample32 - sample12;  
      float mag_sample = sqrt( diff_x*diff_x + diff_y*diff_y );  
      float grad_sample = atan( diff_y / diff_x );  
      if(grad_sample == PI)
      {  
	grad_sample = -PI;
      }  
      // Compute the weighting for the x and y dimensions.  
      float *x_wght=(float *) malloc( GridSpacing * GridSpacing * sizeof(float));  
      float *y_wght=(float *) malloc( GridSpacing * GridSpacing * sizeof(float));  
      float *pos_wght=(float *) malloc( 8*GridSpacing * GridSpacing * sizeof(float));;  
      for (m=0;m<32;++m,++m)  
      {  
	float x=featcenter[m];  
	float y=featcenter[m+1];  
	x_wght[m/2] = fmaxf(1 - (fabs(x - x_sample)*1.0/GridSpacing), 0);  
	y_wght[m/2] = fmaxf(1 - (fabs(y - y_sample)*1.0/GridSpacing), 0);   
	
      }  
      for ( m=0;m<16;++m)
      {  
	for (int n=0;n<8;++n)
	{  
	  pos_wght[m*8+n]=x_wght[m]*y_wght[m];
	}
      }
      free(x_wght);  
      free(y_wght);  
 
      float diff[8],orient_wght[128];  
      for ( m=0;m<8;++m)  
      {   
	float angle = grad_sample-(p->ori)-orient_angles[m]+PI;  
	float temp = angle / (2.0 * PI);  
	angle -= (int)(temp) * (2.0 * PI);  
	diff[m]= angle - PI;  
      }  
      // Compute the gaussian weighting.  
      float x=p->sx;  
      float y=p->sy;  
      float g = exp(-((x_sample-x)*(x_sample-x)+(y_sample-y)*(y_sample-y))/(2*feat_window*feat_window))/(2*CV_PI*feat_window*feat_window);  
      
      for ( m=0;m<128;++m)  
      {  
	orient_wght[m] = fmaxf((1.0 - 1.0*fabs(diff[m%8])/orient_bin_spacing),0);  
	feat_desc[m] = feat_desc[m] + orient_wght[m]*pos_wght[m]*g*mag_sample;  
      }  
      free(pos_wght);     
    }  
    free(feat);  
    free(featcenter);  
    float norm=get_vec_norm( feat_desc, 128);  
    for (m=0;m<128;m++)  
    {  
      feat_desc[m]/=norm;  
      if (feat_desc[m]>0.2)
      {  
	feat_desc[m]=0.2;
      }  
    }  
    norm=get_vec_norm( feat_desc, 128);  
    for ( m=0;m<128;m++)  
    {  
      feat_desc[m]/=norm;  
      //printf("%f  ",feat_desc[m]);    
    }  
    //printf("\n");  
    p->descrip = feat_desc;  
    p=p->next;  
  }  
  free(feat_grid);  
  free(feat_samples); 
}

keypoint sift_desc(mat_c* src)
{
  mat_c* image1Mat = NULL;
  mat_c* tempMat = NULL; 
  int dim;
  image_octaves *Gaussianpyr = NULL;  
  int rows,cols;
  keypoint rt;

  image1Mat = malloc_mat(src->rows, src->cols, 1, MAT_F2D);
  convtp_mat(src, image1Mat);
  conv_scale_mat(image1Mat, 1.0f/255.0f);

  dim = (image1Mat->rows > image1Mat->cols ? image1Mat->cols : image1Mat->rows);  
  //numoctaves = (int) (log((double) dim) / log(2.0)) - 2;   
  //numoctaves = (numoctaves > MAXOCTAVES ? MAXOCTAVES : numoctaves);
  numoctaves = MAXOCTAVES;
  //step 1
  tempMat = scale_init_image(image1Mat);
  //step 2 
  Gaussianpyr = build_gaussian_octaves(tempMat);
  //step 3
  int keycount=detect_keypoint(numoctaves, Gaussianpyr);  
  printf("the keypoints number are %d ;\n", keycount);
  //step 4  
  compute_grad_direcand_mag(numoctaves, Gaussianpyr);
  assign_the_main_orientation( numoctaves, Gaussianpyr,mag_pyr,grad_pyr); 
  //step 5
  extract_feature_descriptors( numoctaves, Gaussianpyr);

  free_mat(image1Mat);
  free_mat(tempMat);
  
  rt = copy_keypoint(keyDescriptors);

  free_keypoint(&keypoints);
  free_keypoint(&keyDescriptors);

  free_image_octaves(DOGoctaves, MAXOCTAVES, SCALESPEROCTAVE + 2);
  free_image_octaves(mag_pyr, MAXOCTAVES, SCALESPEROCTAVE);
  free_image_octaves(grad_pyr, MAXOCTAVES, SCALESPEROCTAVE);
  free_image_octaves(Gaussianpyr, MAXOCTAVES, SCALESPEROCTAVE + 3);


  return rt;
}

int sift_match(keypoint src1, keypoint src2)
{
  int ismatch;
  float mvar, minvar;
  int mcont = 0;
  keypoint s1;
  keypoint s2;
  keypoint min;

  s1 = src1;
  while(s1)
  {
    minvar = MATCH_THRESHOLD;
    ismatch = 0;
    s2 = src2;
    while(s2)
    {
      mvar = get_e_distence(s1->descrip, s2->descrip , LEN);
    
      if(mvar < minvar)
      {
	minvar = mvar;
	ismatch++;
	s1->match = s2;
      }
      s2 = s2->next;
    }
    printf("mvar: %f\n", minvar);
    if(0 == ismatch)
    {
      s1->match = NULL;
    }
    else
    {
      mcont++;
    }
    s1 = s1->next;
  }

  s1 = src2;
  while(s1)
  {
    minvar = MATCH_THRESHOLD;
    ismatch = 0;
    s2 = src1;
    while(s2)
    {
      mvar = get_e_distence(s1->descrip, s2->descrip , LEN);
    
      if(mvar < minvar)
      {
	minvar = mvar;
	ismatch++;
	min = s2;
      }
      s2 = s2->next;
    }
    printf("mvar: %f\n", minvar);
    if(0 == ismatch)
    {
      s1->match = NULL;
    }
    else
    {
      if(s1 != min->match)
      {
	printf("Delete!!!\n");
	mcont--;
	min->match = NULL;
	s1->match = NULL;
      }
      else
      {
	s1->match = min;
      }
    }
    s1 = s1->next;
  }
  return mcont;
}

