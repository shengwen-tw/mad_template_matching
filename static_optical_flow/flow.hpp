#ifndef __FLOW_H__
#define __FLOW_H__

typedef struct {
	/* The point matched from old frame */
	struct {
		int x;
		int y;
	} match_point;

	//match point distance between two frame
	float match_dist;
} flow_t;

#endif
