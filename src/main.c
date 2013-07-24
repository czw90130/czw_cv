#include "config.h"

int main()
{
  cv::Mat img1;
  cv::Mat img2;
  cv::Mat out1; 
  cv::Mat out2;

  mat_c *pmat1; 
  mat_c *pgray1;
  keypoint kps1;

  mat_c *pmat2; 
  mat_c *pgray2;
  keypoint kps2;

  mat_c *lift;

  int mt = 0;
  /*
  img1=cvLoadImage("ref.bmp");
  pmat1 = malloc_mat(img1.rows, img1.cols, img1.channels(), MAT_UC2D);
  pgray1 = malloc_mat(img1.rows, img1.cols, 1, MAT_UC2D);

  img2=cvLoadImage("test2.bmp");
  pmat2 = malloc_mat(img2.rows, img2.cols, img2.channels(), MAT_UC2D);
  pgray2 = malloc_mat(img2.rows, img2.cols, 1, MAT_UC2D);

  imread_cv(img1, pmat1);
  rgb2gray(pmat1, pgray1);

  imread_cv(img2, pmat2);
  rgb2gray(pmat2, pgray2);
  
  canny_edge(pgray);
  kps1 = sift_desc(pgray1);
  kps2 = sift_desc(pgray2);

  mt = sift_match(kps1, kps2);
  printf("%d points matched!!\n", mt);
  
  out1 = imwrite_cv(pmat1);
  out2 = imwrite_cv(pmat2);
  
  display_keypoint_location(out2, kps2);

  free_keypoint(&kps1);
  free_mat(pmat1);
  free_mat(pgray1);

  free_keypoint(&kps2);
  free_mat(pmat2);
  free_mat(pgray2);
  */

  img1=cvLoadImage("test2.bmp");

  pmat1 = malloc_mat(img1.rows, img1.cols, img1.channels(), MAT_UC2D);
  pgray1 = malloc_mat(img1.rows, img1.cols, 1, MAT_UC2D);

  imread_cv(img1, pmat1);
  rgb2gray(pmat1, pgray1);

  //cv::namedWindow("Ref Window");
  
  //cv::imshow("Ref Window", img1);

  lift = malloc_mat(img1.rows/2, img1.cols/2, 1, MAT_UC2D);

  liftwave_low_2d(pgray1, lift);

  canny_edge(lift);

  pmat2 = copy_mat(lift);

  kps1 = hough(lift);

  kps2 = kps1;
  while(kps2)
  {
    printf("x: %f, y: %f, ru: %f, theta: %f\n", kps2->col, kps2->row, kps2->mag, kps2->ori);
    kps2 = kps2->next;
  }

  out1 = imwrite_cv(pmat2);

  display_keypoint_line(out1, kps1);

  free_mat(pmat1);
  free_mat(pmat2);
  free_mat(pgray1);
  free_mat(lift);

  free_keypoint(&kps1);

  cv::namedWindow("Ref Window");
  
  cv::imshow("Ref Window", out1);
  cv::waitKey(0);
  
  
  return 0;
}
