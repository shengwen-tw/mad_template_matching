#include <iostream>

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#include "opencv2/opencv.hpp"

#include "flow.hpp"

using namespace std;
using namespace cv;

#define SAMPLE_RATE 5                      //down sampling factor = 1/5

#define FRAME_SIZE (640 / SAMPLE_RATE)     //640x640 -> 128x128
#define TEMPLATE_SIZE 8                    //8x8
#define SEARCH_AREA 24                     //24x24
#define SUBAREA_SEMI_LEN (SEARCH_AREA / 2) //12

#define TEMPLATE_NUMBER (FRAME_SIZE - TEMPLATE_SIZE + 1) //number 8x8 template split from the frame

#define MAD_THRESHOLD 255 //only accept mad when the value is lower than this threshold

cv::Mat mat_frame1_img;
cv::Mat mat_frame2_img;

uint8_t frame1_image[FRAME_SIZE][FRAME_SIZE] = {0}; //previous frame
uint8_t frame2_image[FRAME_SIZE][FRAME_SIZE] = {0}; //next image

flow_t flow_info[TEMPLATE_NUMBER][TEMPLATE_NUMBER];

void read_image()
{
	mat_frame1_img = imread("frame1.jpg", CV_8UC1);
	mat_frame2_img = imread("frame2.jpg", CV_8UC1);

	/* pixel binning */
	cv::resize(mat_frame1_img, mat_frame1_img, cv::Size(FRAME_SIZE, FRAME_SIZE), CV_INTER_LINEAR);
	cv::resize(mat_frame2_img, mat_frame2_img, cv::Size(FRAME_SIZE, FRAME_SIZE), CV_INTER_LINEAR);

	/* do down sampling and save to uint8_t array */
	for(int i = 0; i < FRAME_SIZE; i++) {
		for(int j = 0; j < FRAME_SIZE; j++) {
			frame1_image[i][j] =
				mat_frame1_img.at<uint8_t>(Point(j, i));
		}
	}

	for(int i = 0; i < FRAME_SIZE; i++) {
		for(int j = 0; j < FRAME_SIZE; j++) {
			frame2_image[i][j] =
				mat_frame2_img.at<uint8_t>(Point(j, i));
		}
	}

	cv::resize(mat_frame1_img, mat_frame1_img,
		cv::Size(FRAME_SIZE * SAMPLE_RATE, FRAME_SIZE * SAMPLE_RATE));
	cv::resize(mat_frame2_img, mat_frame2_img,
		cv::Size(FRAME_SIZE * SAMPLE_RATE, FRAME_SIZE * SAMPLE_RATE));

	/* convert back to rgb color space (represent grey scale in rgb) */
	cv::cvtColor(mat_frame1_img, mat_frame1_img, CV_GRAY2BGR);
	cv::cvtColor(mat_frame2_img, mat_frame2_img, CV_GRAY2BGR);
}

/* calculate mean absolute difference of two 8X8 images */
uint8_t mean_abs_diff(uint8_t *search_area, uint8_t *search_template)
{
	uint32_t mad = 0;

	int i, j;
	for(i = 0; i < TEMPLATE_SIZE; i++) {
		for(j = 0; j < TEMPLATE_SIZE; j++) {
			mad += abs(
				search_template[i * FRAME_SIZE + j] -
				search_area[i * FRAME_SIZE + j]
			);
		}
	}

	mad /= TEMPLATE_SIZE * TEMPLATE_SIZE;

	return (uint8_t)mad;
}

/* calculate 8x8 template's mad value on a 16x16 search subarea */
bool calculate_subarea_mad(uint8_t *search_area, uint8_t *search_template,
	int *match_x, int *match_y)
{
	int _match_x = 0, _match_y = 0;
	int search_size = SEARCH_AREA - TEMPLATE_SIZE + 1;
	uint8_t min_mad = 255, mad; //initial rest

	int i, j;
	for(i = 0; i < search_size; i++) {
		for(j = 0; j < search_size; j++) {
			mad = mean_abs_diff(
				&search_area[i * FRAME_SIZE + j],
				&search_template[i * FRAME_SIZE + j]
			);

			if(mad < min_mad) {
				_match_x = i;
				_match_y = j;
				min_mad = mad;
			}
		}
	}

	if(min_mad > MAD_THRESHOLD) {
		return false;
	}

	/* position on the subarea, not the whole frame! */
	*match_x = _match_x;
	*match_y = _match_y;

	return true;
}

/* calculate 8x8 template's mad value on the whole frame */
bool calculate_mad_full_frame(uint8_t *frame, uint8_t *search_template,
	int *match_x, int *match_y)
{
	int _match_x = 0, _match_y = 0;
	uint8_t min_mad = 255, mad; //initial rest

	int i, j;
	for(i = 0; i < TEMPLATE_NUMBER; i++) {
		for(j = 0; j < TEMPLATE_NUMBER; j++) {
			mad = mean_abs_diff(
				&frame[i * FRAME_SIZE + j],
				&search_template[i * FRAME_SIZE + j]
			);

			if(mad < min_mad) {
				_match_x = i;
				_match_y = j;
				min_mad = mad;
			}
		}
	}

	if(min_mad > MAD_THRESHOLD) {
		return false;
	}

	/* position on the subarea, not the whole frame! */
	*match_x = _match_x;
	*match_y = _match_y;

	return true;
}

/* match all shift feature points on two frames */
void match_feature_points(uint8_t *last_frame, uint8_t *curr_frame)
{
	bool match;
	int match_x = 0, match_y = 0;

	int i, j;
	for(i = 0; i < TEMPLATE_NUMBER; i++) {
		for(j = 0; j < TEMPLATE_NUMBER; j++) {
			match = calculate_mad_full_frame(
				&last_frame[i * FRAME_SIZE + j],
				&curr_frame[i * FRAME_SIZE + j],
				&match_x, &match_y
			);

			if(match == true) {
				flow_info[i][j].match_point.x = match_x;
				flow_info[i][j].match_point.y = match_y;
				flow_info[i][j].no_match_point = false;
				flow_info[i][j].match_dist = sqrt(
					(match_x - i * SAMPLE_RATE) *
					(match_x - i * SAMPLE_RATE) +
					(match_y - j * SAMPLE_RATE) *
					(match_y - j * SAMPLE_RATE)
				);
			} else {
				flow_info[i][j].no_match_point = true;
			}
		}
	}
}

void match_point_visualize(cv::Mat& frame)
{
	int x;
	int y;

	for(int i = 0; i < TEMPLATE_NUMBER; i++) {
		for(int j = 0; j < TEMPLATE_NUMBER; j++) {
			if(flow_info[i][j].no_match_point == true) {
				continue;
			}

			if(flow_info[i][j].match_dist < 20) {
				continue;	
			}

			x = flow_info[i][j].match_point.x * SAMPLE_RATE;
			y = flow_info[i][j].match_point.y * SAMPLE_RATE;

			cv::circle(frame, Point(x, y), 1, Scalar(0, 255, 0), 2, CV_AA, 0);
		}
	}
}

void flow_visualize(cv::Mat& frame)
{
}

int main()
{
	read_image();

	match_feature_points(&frame1_image[0][0], &frame2_image[0][0]);

	//flow_visualize(mat_frame1_img);
	//match_point_visualize(mat_frame1_img);

	cv::imshow("frame2", mat_frame2_img);
	cv::imshow("frame1", mat_frame1_img);

	cv::waitKey(0);

	return 0;
}
