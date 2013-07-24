#include "hough.h"

int scan_subarea(mat_c* src, int si, int sj, float* ru, float* theta)
{
  int i, j, k;

  int hs = si+SCAN_AREA/2 > src->cols ? src->cols : si+SCAN_AREA/2;
  int vs = sj+SCAN_AREA/2 > src->rows ? src->rows : sj+SCAN_AREA/2;

  int ssi = si-SCAN_AREA/2 < 0 ? 0 : si-SCAN_AREA/2;
  int ssj = sj-SCAN_AREA/2 < 0 ? 0 : sj-SCAN_AREA/2;

  //printf("si: %d sj: %d ssi: %d, ssj: %d, hs: %d, vs: %d\n",si, sj, ssi, ssj, hs, vs);

  int count = 0;
  float rus[SCAN_AREA*SCAN_AREA-1];
  float thetas[SCAN_AREA*SCAN_AREA-1];
  int simm[SCAN_AREA*SCAN_AREA-1];

  int maxn = SCAN_AREA_THO;
  int idx = 99;

  for(j=ssj;j<vs;j++)
  {
    for(i=ssi;i<hs;i++)
    {
      int flg = 1;

      if(i == si && j == sj)
      {	
	//printf(" oo");
	continue;
      }
      
      if(50 < src->data.ptr[j][i])
      {
	thetas[count] = atan2((float)(i-si), (float)(j-sj));
	if(thetas[count] > 0)
	{
	  thetas[count] = PI - thetas[count];
	}
	else
	{
	  thetas[count] = -thetas[count];
	}
	rus[count] = i*cos(thetas[count]) + j*sin(thetas[count]);

	//printf("[%d] thetas: %f, rus: %f\n", count, thetas[count], rus[count]);

	simm[count] = 1;
	flg = 1;
	for(k=0;k<count;k++)
	{
	  if(SCAN_DELTA_THETA > fabs(thetas[count] - thetas[k])
	     && SCAN_DELTA_RU > fabs(rus[count] - rus[k]))
	  {
	    flg = 0;
	    simm[k]++;
	    thetas[k] = (thetas[count] + thetas[k])/2;
	    rus[k] = (rus[count] + rus[k])/2;
	    //printf("[%d] (x: %d, y: %d),  K: %d\n", count, i, j, k);
	    break;
	  }
	}
	
	if(1 == flg)
	{
	  //printf(" xx");  
	  count++;
	}
	/* else */
        /* { */
	/*   printf(" %2d",k); */
	/* } */
      }
      /* else */
      /* { */
      /* 	printf(" --"); */
      /* } */

    }
  //printf("\n");
  }

  for(i=0;i<count;i++)
  {
    if(maxn < simm[i])
    {
      maxn = simm[i];
      idx = i;
    }
  }

  
  if(99 == idx)
  {
    //printf("NO-------------------\n\n");
    return 0;
  }
  else
  {
    *ru = rus[idx];
    *theta = thetas[idx];
    //printf("maxn: %d, idx: %d theta: %f, ru: %f\n\n", maxn, idx, *theta, *ru);

    //cv::waitKey(0);
    return 1;
  }
}

int scan_allarea(mat_c* src, int si, int sj, float ru, float theta, int isdelete)
{
  int i, j;

  int count = 0;

  float ru1, theta1;

  for(j=sj;j<src->rows;j++)
  {
    for(i=0;i<src->cols;i++)
    { 
      if(50 == src->data.ptr[j][i] && 1 == isdelete)
      {
	src->data.ptr[j][i] = 0;
      }
      else if(50 == src->data.ptr[j][i] && 0 == isdelete)
      {
	src->data.ptr[j][i] = 255;
      }
      if(!(si == i && sj == j) && 50 < src->data.ptr[j][i])
      {
	theta1 = atan2((float)(i-si), (float)(j-sj));
	if(theta1 > 0)
	{
	  theta1 = PI - theta1;
	}
	else
	{
	  theta1 = -theta1;
	}
	ru1 = i*cos(theta1) + j*sin(theta1);
	if(SCAN_DELTA_THETA > fabs(theta - theta1)
	   && SCAN_DELTA_RU > fabs(ru - ru1))
	{
	  src->data.ptr[j][i] = 50;
	  count++;
	} // end if    
      }
    }
  }
  //printf("YES!!!!!!!!!!11 %d\n",count);
  return count;
}

keypoint hough(mat_c* src)
{
  int i,j;
  keypoint k, keys;
  float ru, theta;
  int deleteflg = 0;

  /* Allocate memory for the keypoint. */  
  k = (keypoint) malloc(sizeof(struct KeypointSt)); 
  k->next = NULL;
  k->descrip = NULL;
  keys = k;

  for(j=0;j<src->rows;j++)
  {
    for(i=0;i<src->cols;i++)
    {
      if(50 < src->data.ptr[j][i])
      {
	if(1 == scan_subarea(src, i, j, &ru, &theta))
	{
	  if(MIN_POINTS < scan_allarea(src, i, j, ru, theta, deleteflg))
	  {
	    deleteflg = 1;
	    k->ori = theta;
	    k->mag = ru;
	    k->col = i;
	    k->row = j;
	    /* Allocate memory for the keypoint. */ 
	    k = (keypoint) malloc(sizeof(struct KeypointSt)); 
	    k->next = keys;
	    k->descrip = NULL;
	    keys = k;
	  }
	  else
	  {
	    deleteflg = 0;
	  }
	}
	src->data.ptr[j][i] = 0;	
      }
    }
  }
  
  return keys;
}
