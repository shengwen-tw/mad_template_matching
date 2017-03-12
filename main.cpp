#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#define SAMPLE_RATE 32
#define FULL_IMG_SIZE (640 / SAMPLE_RATE) //pixel
#define SEARCH_IMG_SIZE (64 / SAMPLE_RATE)
#define MAD_MAP_SIZE (FULL_IMG_SIZE - SEARCH_IMG_SIZE + 1)

uint8_t full_image[FULL_IMG_SIZE][FULL_IMG_SIZE] = {0};
uint8_t search_image[SEARCH_IMG_SIZE][SEARCH_IMG_SIZE] = {0};
uint8_t mad_map[MAD_MAP_SIZE][MAD_MAP_SIZE] = {0};

cv::Mat mat_full_img;
cv::Mat mat_search_img;

void read_image()
{
	/* Read my cute polandball pictures */
	mat_full_img = imread("polandball.jpg");
	mat_search_img = imread("search.jpg");

	/* convert to grey scale but represent in rgb space */
	cv::cvtColor(mat_full_img, mat_full_img, CV_BGR2GRAY);
	cv::cvtColor(mat_full_img, mat_full_img, CV_GRAY2BGR);
	cv::cvtColor(mat_search_img, mat_search_img, CV_BGR2GRAY);
	cv::cvtColor(mat_search_img, mat_search_img, CV_GRAY2BGR);

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

uint8_t calculate_mad(uint8_t *full_img, uint8_t *search_img)
{
	uint32_t mad = 0;

	int i, j;
	for(i = 0; i < SEARCH_IMG_SIZE; i++) {
		for(j = 0; j < SEARCH_IMG_SIZE; j++) {
			mad += abs(
				search_img[i * SEARCH_IMG_SIZE + j] -
				full_img[i * SEARCH_IMG_SIZE + j]
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

	min_mad = mad_map[0] = calculate_mad(&full_img[0], search_img);

	int i, j;
	for(i = 1; i < MAD_MAP_SIZE; i++) {
		for(j = 1 ; j < MAD_MAP_SIZE; j++) {
			mad = mad_map[i * MAD_MAP_SIZE + j] =
				calculate_mad(&full_img[i * MAD_MAP_SIZE + j], search_img);

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

int main()
{
	read_image();

	printf("full image down sample by factor 1/%d:\n", SAMPLE_RATE);
	print_picture_value(&full_image[0][0], FULL_IMG_SIZE);
	printf("search image down sample by factor 1/%d:\n", SAMPLE_RATE);
	print_picture_value(&search_image[0][0], SEARCH_IMG_SIZE);


	int match_i = 0, match_j = 0;
	printf("mean absolute difference of two images\n");
	calculate_image_mad(&mad_map[0][0], &full_image[0][0], &search_image[0][0], &match_i, &match_j);
	print_picture_value(&mad_map[0][0], MAD_MAP_SIZE);
	printf("Best matched position:(%d, %d)\n", match_i, match_j);

	cv::circle(mat_full_img, Point(match_i * SAMPLE_RATE, match_j * SAMPLE_RATE), 3, Scalar(0, 0, 255), 2, CV_AA, 0);
	cv::imshow("full image", mat_full_img);
	cv::imshow("search image", mat_search_img);

	cv::waitKey(0);

	return 0;
}
