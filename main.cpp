#include <iostream>
#include <stdio.h>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#define SAMPLE_RATE 32
#define FULL_IMG_SIZE (640 / SAMPLE_RATE) //pixel
#define SEARCH_IMG_SIZE (64 / SAMPLE_RATE)

uint8_t full_image[FULL_IMG_SIZE][FULL_IMG_SIZE] = {0};
uint8_t search_image[SEARCH_IMG_SIZE][SEARCH_IMG_SIZE] = {0};

void read_image()
{
	/* Read my cute polandball pictures */
	cv::Mat mat_full_img = imread("polandball.jpg");
	cv::Mat mat_search_img = imread("search.jpg");

	/* convert to grey scale but represent in rgb space */
	cv::cvtColor(mat_full_img, mat_full_img, CV_BGR2GRAY);
	cv::cvtColor(mat_full_img, mat_full_img, CV_GRAY2BGR);
	cv::cvtColor(mat_search_img, mat_search_img, CV_BGR2GRAY);
	cv::cvtColor(mat_search_img, mat_search_img, CV_GRAY2BGR);

	cv::imshow("full image", mat_full_img);
	cv::imshow("search image", mat_search_img);

	/* Do down sampling and save to uint8_t array */
	for(int i = 0; i < FULL_IMG_SIZE; i++) {
		for(int j = 0; j < FULL_IMG_SIZE; j++) {
			full_image[i][j] =
				mat_full_img.at<uint8_t>(Point(i * SAMPLE_RATE, j * SAMPLE_RATE));
		}
	}

	for(int i = 0; i < SEARCH_IMG_SIZE; i++) {
		for(int j = 0; j < SEARCH_IMG_SIZE; j++) {
			search_image[i][j] =
				mat_search_img.at<uint8_t>(Point(i * SAMPLE_RATE, j * SAMPLE_RATE));
		}
	}
}

void print_picture_value(uint8_t *image_arr, int size)
{
	for(int i = 0; i < size; i++) {
		for(int j = 0; j < size; j++) {
			printf("%03u ", image_arr[i * size + j]);
		}
		printf("\n");
	}
	printf("\n");
}

int main()
{
	read_image();

	printf("full image down sample by factor 1/%d:\n", SAMPLE_RATE);
	print_picture_value(&full_image[0][0], FULL_IMG_SIZE);
	printf("search image down sample by factor 1/%d:\n", SAMPLE_RATE);
	print_picture_value(&search_image[0][0], SEARCH_IMG_SIZE);

	cv::waitKey(0);

	return 0;
}
