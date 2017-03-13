#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "opencv2/opencv.hpp"

#include "flow.hpp"

using namespace std;
using namespace cv;

#define SAMPLE_RATE 16
#define FRAME_SIZE (480 / SAMPLE_RATE) //pixel
#define SEARCH_SIZE (64 / SAMPLE_RATE)
#define MAD_MAP_SIZE (FRAME_SIZE - SEARCH_SIZE + 1)

cv::Mat mat_frame1_img;
cv::Mat mat_frame2_img;

uint8_t frame1_image[FRAME_SIZE][FRAME_SIZE] = {0}; //previous frame
uint8_t frame2_image[FRAME_SIZE][FRAME_SIZE] = {0}; //next image

uint8_t mad_map[MAD_MAP_SIZE][MAD_MAP_SIZE] = {0}; //absolute difference map

flow_t flow_info[FRAME_SIZE][FRAME_SIZE];

void read_image()
{
	mat_frame1_img = imread("frame1.jpg", CV_8UC1);
	mat_frame2_img = imread("frame2.jpg", CV_8UC1);

	/* Do down sampling and save to uint8_t array */
	for(int i = 0; i < FRAME_SIZE; i++) {
		for(int j = 0; j < FRAME_SIZE; j++) {
			frame1_image[i][j] =
				mat_frame1_img.at<uint8_t>(Point(j * SAMPLE_RATE, i * SAMPLE_RATE));
		}
	}

	for(int i = 0; i < FRAME_SIZE; i++) {
		for(int j = 0; j < FRAME_SIZE; j++) {
			frame2_image[i][j] =
				mat_frame2_img.at<uint8_t>(Point(j * SAMPLE_RATE, i * SAMPLE_RATE));
		}
	}

	/* Convert back to rgb color space (represent grey scale in rgb) */
	cv::cvtColor(mat_frame1_img, mat_frame1_img, CV_GRAY2BGR);
	cv::cvtColor(mat_frame2_img, mat_frame2_img, CV_GRAY2BGR);
}

uint8_t calculate_mad(uint8_t *frame, uint8_t *search_img)
{
	uint32_t mad = 0;

	int i, j;
	for(i = 0; i < FRAME_SIZE; i++) {
		for(j = 0; j < FRAME_SIZE; j++) {
			mad += abs(
				search_img[i * FRAME_SIZE + j] -
				frame[i * FRAME_SIZE + j]
			);
		}
	}

	mad /= FRAME_SIZE * FRAME_SIZE;

	return (uint8_t)mad;
}

void calculate_image_mad(uint8_t *mad_map, uint8_t *frame, uint8_t *search_img,
	int *match_i, int *match_j)
{
	int min_mad, mad;
	int _match_i = 0, _match_j = 0;

	min_mad = 255; //Initial rest

	int i, j;
	for(i = 0; i < MAD_MAP_SIZE; i++) {
		for(j = 0 ; j < MAD_MAP_SIZE; j++) {
			mad = mad_map[i * MAD_MAP_SIZE + j] =
				calculate_mad(&frame[i * FRAME_SIZE + j], search_img);

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

void calculate_match_points()
{
	int match_x = 0, match_y = 0;

	int i, j;
	for(i = 0; i < FRAME_SIZE; i++) {
		for(j = 0; j < FRAME_SIZE; j++) {
			/* Crop small area from frame1 as search template and caluclate the
			   absolute difference from frame2  */
			calculate_image_mad(&mad_map[0][0], &frame2_image[0][0],
				&frame1_image[i][j], &match_x, &match_y);

			flow_info[i][j].match_point.x = match_x;
			flow_info[i][j].match_point.y = match_y;

			flow_info[i][j].match_dist = sqrt(
				(match_x - i) * (match_x - i) +
				(match_y - j) * (match_y - j)
			);
		}
	}
}

void show_down_sample_image()
{
	cv::Mat down_full_img = cv::Mat(FRAME_SIZE, FRAME_SIZE, CV_8UC1, frame1_image);
	cv::imshow("down sample - full image", down_full_img);
	cv::Mat down_search_img = cv::Mat(FRAME_SIZE, FRAME_SIZE, CV_8UC1, frame2_image);
	cv::imshow("down sample - search image", down_search_img);
}

void show_absolute_difference_map()
{
	cv::Mat ad_img = cv::Mat(MAD_MAP_SIZE, MAD_MAP_SIZE, CV_8UC1, mad_map);
	cv::imshow("absolute difference map", ad_img);
}

void match_point_visualize(cv::Mat& frame)
{
	int x = 0, y = 0;
	for(int i = 0; i < FRAME_SIZE; i++) {
		for(int j = 0; j < FRAME_SIZE; j++) {
			/* Thresholding */
			if(flow_info[i][j].match_dist < 30.0f) {
				continue;
			}

			x = i * SAMPLE_RATE;
			y = j * SAMPLE_RATE;
			cv::circle(frame, Point(x, y), 1, Scalar(0, 255, 0), 2, CV_AA, 0);
		}
	}
}

void flow_visualize(cv::Mat& frame)
{
	int x = 0, y = 0;
	for(int i = 0; i < FRAME_SIZE; i++) {
		for(int j = 0; j < FRAME_SIZE; j++) {
			x = i * SAMPLE_RATE;
			y = j * SAMPLE_RATE;
			cv::circle(frame, Point(x, y), 1, Scalar(0, 255, 0), 2, CV_AA, 0);
		}
	}
}

int main()
{
	read_image();

	calculate_match_points();

	//flow_visualize(mat_frame1_img);
	match_point_visualize(mat_frame1_img);

	cv::imshow("frame2", mat_frame2_img);
	cv::imshow("frame1", mat_frame1_img);

	//show_match_probability_map();

	cv::waitKey(0);

	return 0;
}
