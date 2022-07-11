//
// Created by dev on 7/8/22.
//

#ifndef COURSE_RENDERER_KERNEL_CUH
#define COURSE_RENDERER_KERNEL_CUH

#include "model/inc/model.h"
#include <helper_functions.h>

struct Image {
	uint *pixels;
	int width;
	int height;

	__device__ void set(int x, int y, uint value) const {
		if (x < 0 or x >= width) return;
		if (y < 0 or y >= height) return;
		pixels[y * width + x] = value;
	}

	__device__ void add(int x, int y, uint value) const {
		if (x < 0 or x >= width) return;
		if (y < 0 or y >= height) return;
		pixels[y * width + x] += value;
	}
};


double main_cuda_launch(Image &image,
                        StopWatchInterface *timer);

#endif//COURSE_RENDERER_KERNEL_CUH
