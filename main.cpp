#include <iostream>
#include <stdio.h>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#define FULL_IMG_SIZE 640 //640X640
#define SEARCH_IMG_SIZE 64 //64X64

uint8_t full_image[FULL_IMG_SIZE][FULL_IMG_SIZE];
uint8_t search_image[SEARCH_IMG_SIZE][SEARCH_IMG_SIZE];

void read_image()
{
	cv::Mat mat_full_img = imread("polandball.jpg");
	cv::Mat mat_search_img = imread("search.jpg");

	/* convert to grey scale but represent in rgb space */
	cv::cvtColor(mat_full_img, mat_full_img, CV_BGR2GRAY);
	cv::cvtColor(mat_full_img, mat_full_img, CV_GRAY2BGR);
	cv::cvtColor(mat_search_img, mat_search_img, CV_BGR2GRAY);
	cv::cvtColor(mat_search_img, mat_search_img, CV_GRAY2BGR);

	cv::imshow("full image", mat_full_img);
	cv::imshow("search image", mat_search_img);
}

int main()
{
	read_image();
	cv::waitKey(0);

	return 0;
}
