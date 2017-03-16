#include <iostream>

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#include "opencv2/opencv.hpp"

#include "flow.hpp"

using namespace std;
using namespace cv;

#define SAMPLE_RATE 10                     //down sampling factor = 1/5

#define FRAME_SIZE (640 / SAMPLE_RATE)     //640x640 -> 128x128
#define TEMPLATE_SIZE 8                    //8x8
#define SEARCH_AREA 24                     //24x24

#define TEMPLATE_FRAME_OFFSET ((SEARCH_AREA - TEMPLATE_SIZE) / 2)
#define FLOW_NUMBER (FRAME_SIZE - (TEMPLATE_FRAME_OFFSET * 2))

#define MAD_THRESHOLD 255 //only accept mad when the value is lower than this threshold

cv::Mat mat_frame1_img;
cv::Mat mat_frame2_img;

uint8_t frame1_image[FRAME_SIZE][FRAME_SIZE] = {0}; //previous frame
uint8_t frame2_image[FRAME_SIZE][FRAME_SIZE] = {0}; //next image

flow_t flow_info[FLOW_NUMBER][FLOW_NUMBER];

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

bool calculate_subarea_mad(uint8_t *search_subarea, uint8_t *search_template,
	int *match_x, int *match_y)
{
	int _match_x = 0, _match_y = 0;
	uint8_t min_mad = 255, mad; //initial rest

	int i, j;
	for(i = 0; i < SEARCH_AREA; i++) {
		for(j = 0; j < SEARCH_AREA; j++) {
			mad = mean_abs_diff(
				&search_subarea[i * FRAME_SIZE + j],
				&search_template[0]
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

	/* The match point position on local subarea, need to shift
	 * with respect to the  local center */
	*match_x = _match_x - TEMPLATE_FRAME_OFFSET;
	*match_y = _match_y - TEMPLATE_FRAME_OFFSET;

	//printf("(%d, %d)\n", *match_x, *match_y);

	return true;
}

void match_feature_points(uint8_t *last_frame, uint8_t *curr_frame)
{
	bool match;
	int match_x = 0, match_y = 0;

	int subarea_center_x, subarea_center_y;

	int i, j;
	for(i = 0; i < FLOW_NUMBER; i++) {
		for(j = 0; j < FLOW_NUMBER; j++) {
			subarea_center_x = i + TEMPLATE_FRAME_OFFSET;
			subarea_center_y = j + TEMPLATE_FRAME_OFFSET;

			match = calculate_subarea_mad(
				&last_frame[0],
				&curr_frame[subarea_center_x * FRAME_SIZE + subarea_center_y],
				&match_x, &match_y
			);

			if(match == true) {
#if 1
				if(match_x == -8 || match_y == -8) {
					flow_info[i][j].no_match_point = true;
					continue;
				}
#endif
				flow_info[i][j].match_point.x = i ;//+ match_x;
				flow_info[i][j].match_point.y = j ;//+ match_y;
				flow_info[i][j].no_match_point = false;
				flow_info[i][j].match_dist = sqrtf(
					((float)match_x - (float)i) *
					((float)match_x - (float)i) +
					((float)match_y - (float)j) *
					((float)match_y - (float)j)
				);
			} else {
				flow_info[i][j].no_match_point = true;
			}
		}
	}
}

void match_point_visualize(cv::Mat& frame1, cv::Mat& frame2)
{
	int x;
	int y;

	for(int i = 0; i < FLOW_NUMBER; i++) {
		for(int j = 0; j < FLOW_NUMBER; j++) {
			if(flow_info[i][j].no_match_point == true) {
				continue;
			}


			//printf("match point distance: %f\n", flow_info[i][j].match_dist);
			//printf("match point location: (%d, %d)\n", x, y);

			x = (flow_info[i][j].match_point.y + TEMPLATE_FRAME_OFFSET) * SAMPLE_RATE;
			y = (flow_info[i][j].match_point.x + TEMPLATE_FRAME_OFFSET) * SAMPLE_RATE;
			cv::circle(frame1, Point(x, y), 1, Scalar(0, 255, 0), 2, CV_AA, 0);

			x = (j + TEMPLATE_FRAME_OFFSET) * SAMPLE_RATE;
			y = (i + TEMPLATE_FRAME_OFFSET) * SAMPLE_RATE;
			cv::circle(frame2, Point(x, y), 1, Scalar(0, 0, 255), 2, CV_AA, 0);
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
	match_point_visualize(mat_frame2_img, mat_frame1_img);
	//match_point_visualize(mat_frame2_img);

	cv::imshow("frame2", mat_frame2_img);
	cv::imshow("frame1", mat_frame1_img);

	cv::waitKey(0);

	return 0;
}
