#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#define SAMPLE_RATE 1
#define FULL_IMG_SIZE (480 / SAMPLE_RATE) //pixel
#define SEARCH_IMG_SIZE (37 / SAMPLE_RATE)
#define MAD_MAP_SIZE (FULL_IMG_SIZE - SEARCH_IMG_SIZE + 1)

uint8_t full_image[FULL_IMG_SIZE][FULL_IMG_SIZE] = {0}; //image to search on
uint8_t search_image[SEARCH_IMG_SIZE][SEARCH_IMG_SIZE] = {0}; //template image
uint8_t mad_map[MAD_MAP_SIZE][MAD_MAP_SIZE] = {0}; //mean absolute difference map
uint8_t match_prob_map[MAD_MAP_SIZE][MAD_MAP_SIZE] = {0}; //match probability map

cv::Mat mat_full_img;
cv::Mat mat_search_img;

int read_camera_grey(cv::VideoCapture& camera)
{
	if(!camera.read(mat_full_img)) {
		return 1; //failed to read image from camera
	}

	mat_full_img = mat_full_img(Rect(80, 0, 480, 480));

	/* convert to grey scale  */
	cv::cvtColor(mat_full_img, mat_full_img, CV_BGR2GRAY);

	/* Do down sampling and save to uint8_t array */
	for(int i = 0; i < FULL_IMG_SIZE; i++) {
		for(int j = 0; j < FULL_IMG_SIZE; j++) {
			full_image[i][j] =
				mat_full_img.at<uint8_t>(Point(j * SAMPLE_RATE, i * SAMPLE_RATE));
		}
	}

	/* Convert back to rgb color space (represent grey scale in rgb) */
	cv::cvtColor(mat_full_img, mat_full_img, CV_GRAY2BGR);

	return 0;
}

void read_search_template()
{
	mat_search_img = imread("search.jpg");

	/* convert to grey scale  */
	cv::cvtColor(mat_search_img, mat_search_img, CV_BGR2GRAY);

	for(int i = 0; i < SEARCH_IMG_SIZE; i++) {
		for(int j = 0; j < SEARCH_IMG_SIZE; j++) {
			search_image[i][j] =
				mat_search_img.at<uint8_t>(Point(j * SAMPLE_RATE, i * SAMPLE_RATE));
		}
	}

	/* Convert back to rgb color space (represent grey scale in rgb) */
	cv::cvtColor(mat_search_img, mat_search_img, CV_GRAY2BGR);
}

uint8_t calculate_mad(uint8_t *full_img, uint8_t *search_img)
{
	uint32_t mad = 0;

	int i, j;
	for(i = 0; i < SEARCH_IMG_SIZE; i++) {
		for(j = 0; j < SEARCH_IMG_SIZE; j++) {
			mad += abs(
				search_img[i * SEARCH_IMG_SIZE + j] -
				full_img[i * FULL_IMG_SIZE + j]
			);
		}
	}

	mad /= SEARCH_IMG_SIZE * SEARCH_IMG_SIZE;

	return (uint8_t)mad;
}

void calculate_image_mad(uint8_t *mad_map, uint8_t *full_img, uint8_t *search_img,
	int *match_i, int *match_j)
{
	int min_mad, mad;
	int _match_i = 0, _match_j = 0;

	min_mad = 255; //Initial rest

	int i, j;
	for(i = 0; i < MAD_MAP_SIZE; i++) {
		for(j = 0 ; j < MAD_MAP_SIZE; j++) {
			mad = mad_map[i * MAD_MAP_SIZE + j] =
				calculate_mad(&full_img[i * FULL_IMG_SIZE + j], search_img);

			if(mad < min_mad) {
				min_mad = mad;
				_match_i = i;
				_match_j = j;		
			}
		}
	}

	*match_i = _match_i;
	*match_j = _match_j;
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

void show_down_sample_image()
{
	cv::Mat down_full_img = cv::Mat(FULL_IMG_SIZE, FULL_IMG_SIZE, CV_8UC1, full_image);
	cv::imshow("down sample - full image", down_full_img);
	cv::Mat down_search_img = cv::Mat(SEARCH_IMG_SIZE, SEARCH_IMG_SIZE, CV_8UC1, search_image);
	cv::imshow("down sample - search image", down_search_img);
}

void show_match_probability_map()
{
	for(int i = 0; i < MAD_MAP_SIZE; i++) {
		for(int j = 0; j < MAD_MAP_SIZE; j++) {
			match_prob_map[i][j] = 255 - mad_map[i][j];
		}
	}

	cv::Mat match_prob_img = cv::Mat(MAD_MAP_SIZE, MAD_MAP_SIZE, CV_8UC1, match_prob_map);
	cv::imshow("match probality map", match_prob_img);
}

void show_absolute_difference_map()
{
	cv::Mat ad_img = cv::Mat(MAD_MAP_SIZE, MAD_MAP_SIZE, CV_8UC1, mad_map);
	cv::imshow("absolute difference map", ad_img);
}

int main()
{
	cv::VideoCapture camera(0);

	if (!camera.isOpened()) {
		return 0;
	}

	read_search_template();

	int match_i = 0, match_j = 0;

	while(!read_camera_grey(camera)) {
		calculate_image_mad(&mad_map[0][0], &full_image[0][0], &search_image[0][0], &match_i, &match_j);

		int x1 = match_j * SAMPLE_RATE;
		int y1 = match_i * SAMPLE_RATE;
		int x2 = (match_j + SEARCH_IMG_SIZE) * SAMPLE_RATE;
		int y2 = (match_i + SEARCH_IMG_SIZE) * SAMPLE_RATE;
		cv::rectangle(mat_full_img, Point(x1, y1), Point(x2, y2), Scalar(0,0,255), 2);

		show_absolute_difference_map();
		//show_match_probability_map();

		cv::imshow("webcam", mat_full_img);
		cv::imshow("search image", mat_search_img);
		cv::waitKey(10);
	}

	return 0;
}
