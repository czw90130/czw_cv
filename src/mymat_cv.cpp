#include "mymat.h"

int imread_cv(cv::Mat mat, mat_c* dst)
{
  int nx, ny, ncol;
  uchar* pcol;
  if(NULL == dst || MAT_UC2D != dst->type)
  {
    return 0;
  }
  ncol = mat.cols*mat.channels();
  for(ny=0; ny<mat.rows; ny++)
  {
    pcol = mat.ptr<uchar>(ny);
    for(nx=0; nx < ncol; nx++)
    {
      dst->data.ptr[ny][nx] = (unsigned char)pcol[nx];
    }
  
  }
  return 1;
}


cv::Mat imwrite_cv(const mat_c* src)
{
  int nx, ny, ncol;
  uchar* pcol;
  cv::Size mat_sz;
  cv::Mat mat;

  mat_sz = cv::Size(src->cols, src->rows);
  switch (src->step /src->cols)
  {
  case 1:
    mat = cv::Mat(mat_sz, CV_8UC1);
    break;

  case 3:
  default:
    mat = cv::Mat(mat_sz, CV_8UC3);
    break;    
  }
  
  if(NULL == src || MAT_UC2D != src->type)
  {
    return mat;
  }
  ncol = mat.cols*mat.channels();
  for(ny=0; ny<mat.rows; ny++)
  {
    pcol = mat.ptr<uchar>(ny);
    for(nx=0; nx < ncol; nx++)
    {
      pcol[nx] = src->data.ptr[ny][nx];
    }
  
  }
  return mat;
}

void display_keypoint_location(cv::Mat image, keypoint keypoints)  
{  
   
  int i = 0;
  keypoint p = keypoints; 
  while(p)   
  {     
    cv::line( image, cv::Point((int)((p->col)-3),(int)(p->row)),   
	      cv::Point((int)((p->col)+3),(int)(p->row)), CV_RGB(255,255,0),  
	    1, 8, 0 );  
    cv::line( image, cv::Point((int)(p->col),(int)((p->row)-3)),   
	      cv::Point((int)(p->col),(int)((p->row)+3)), CV_RGB(255,255,0),  
	    1, 8, 0 );
    //printf("[%d] x:%d, y:%d\n", ++i, (int)((p->col)), (int)(p->row));
    if(NULL != p->match)
    {

      cv::line( image, cv::Point((int)((p->col)-3),(int)(p->row-3)),   
	      cv::Point((int)((p->col)+3),(int)(p->row+3)), CV_RGB(255,0,0),  
	    1, 8, 0 );  
      cv::line( image, cv::Point((int)(p->col+3),(int)((p->row)-3)),   
		cv::Point((int)(p->col-3),(int)((p->row)+3)), CV_RGB(255,0,0),  
		1, 8, 0 );
      printf("[%d] x:%d, y:%d\n", ++i, (int)((p->col)), (int)(p->row));
    }  
    p=p->next;  
    }   
}

void display_keypoint_line(cv::Mat image, keypoint keypoints)  
{ 

  keypoint p = keypoints->next;
  while(p)
 {
   int x1, y1;
   int x2, y2;

   int x[4], y[4];
   int t;

   cv::line( image, cv::Point((int)((p->col)-3),(int)(p->row)),   
	      cv::Point((int)((p->col)+3),(int)(p->row)), CV_RGB(55,155,155),  
	    1, 8, 0 );  
    cv::line( image, cv::Point((int)(p->col),(int)((p->row)-3)),   
	      cv::Point((int)(p->col),(int)((p->row)+3)), CV_RGB(55,155,155),  
	    1, 8, 0 );

   x[0] = (int)p->mag/cos(p->ori);
   y[0] = 0;
   x[1] = 0;
   y[1] = (int)p->mag/sin(p->ori);
   x[2] = (int)(p->mag-image.rows*sin(p->ori))/cos(p->ori);
   y[2] = image.rows;
   x[3] = image.cols;
   y[3] = (int)(p->mag-image.cols*cos(p->ori))/sin(p->ori);


   for(t=0;t<4;t++)
   {
     if(x[t]>=0 && y[t]>=0)
     {
       x1 = x[t];
       y1 = y[t];
       break;
     }
   }

   for(t=t+1;t<4;t++)
   {
     if(x[t]>=0 && y[t]>=0)
     {
       x2 = x[t];
       y2 = y[t];
       break;
     }
   }

   cv::line( image, cv::Point(x1,y1), cv::Point(x2,y2), 
	     CV_RGB(255,100,155), 1, 8, 0 );  
   p = p->next;   
 }
}
