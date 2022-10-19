//
// Created by dev on 10/4/22.
//

#ifndef COURSE_RENDERER_IMAGE_CUH
#define COURSE_RENDERER_IMAGE_CUH

struct Image {
	int width;
	int height;
	uint *pixels;

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

#endif//COURSE_RENDERER_IMAGE_CUH
