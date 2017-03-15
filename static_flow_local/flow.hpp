#ifndef __FLOW_H__
#define __FLOW_H__

#include <stdbool.h>

typedef struct {
	/* The point matched from old frame */
	struct {
		int x;
		int y;
	} match_point;

	//match point distance between two frame
	float match_dist;

	bool no_match_point;
} flow_t;

#endif
