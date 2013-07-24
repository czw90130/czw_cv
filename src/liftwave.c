#include "liftwave.h"


int liftwave_low_2d(mat_c* src, mat_c* dst)
{
  int i,j,rt;

  mat_c* even1 = malloc_mat(src->rows, src->cols/2, 1, MAT_F2D);
  mat_c* odd1 = malloc_mat(src->rows, src->cols/2, 1, MAT_F2D);

  mat_c* even2 = malloc_mat(even1->rows/2, even1->cols, 1, MAT_F2D);
  mat_c* odd2 = malloc_mat(even1->rows/2, even1->cols, 1, MAT_F2D);

  if(NULL == src || src->step != src->cols)
  {
    return 0;
  }

  for(j=0;j<src->rows;j++)
  {
    for(i=0;i<even1->cols;i++)
    {
      odd1->data.f2d[j][i] = (float)src->data.ptr[j][2*i] - (float)src->data.ptr[j][2*i+1];
      even1->data.f2d[j][i] = (0.5 * odd1->data.f2d[j][i] + (float)src->data.ptr[j][2*i]);
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////


  for(j=0;j<even1->cols;j++)
  {
    for(i=0;i<even2->rows;i++)
    {
      odd2->data.f2d[i][j] = (float)even1->data.f2d[2*i][j] - (float)even1->data.f2d[2*i+1][j];
      even2->data.f2d[i][j] = (0.5 * odd2->data.f2d[i][j] + (float)even1->data.f2d[2*i][j]);

    }
  }
  rt = convtp_mat(even2, dst);
  
  free_mat(odd1);
  free_mat(odd2);
  free_mat(even1);
  free_mat(even2);

  return rt;
}

int liftwave_high_2d(mat_c* src, mat_c* dst)
{
  int i,j,rt;

  mat_c* odd1 = malloc_mat(src->rows, src->cols/2, 1, MAT_F2D);

  mat_c* odd2 = malloc_mat(odd1->rows/2, odd1->cols, 1, MAT_F2D);

  if(NULL == src || src->step != src->cols)
  {
    return 0;
  }

  for(j=0;j<src->rows;j++)
  {
    for(i=1;i<odd1->cols;i++)
    {
      odd1->data.f2d[j][i] = (float)src->data.ptr[j][2*i] - (float)src->data.ptr[j][2*i+1];
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////


  for(j=0;j<odd1->cols;j++)
  {
    for(i=1;i<odd2->rows;i++)
    {
      odd2->data.f2d[i][j] = (float)odd1->data.f2d[2*i][j] -  odd1->data.f2d[2*i+1][j];
    }
  }

  rt = convtp_mat(odd2, dst);

  free_mat(odd1);
  free_mat(odd2);

  return rt;
}
