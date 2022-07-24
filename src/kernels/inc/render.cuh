//
// Created by dev on 7/8/22.
//

#ifndef COURSE_RENDERER_RENDER_CUH
#define COURSE_RENDERER_RENDER_CUH

#include "../../../Common/helper_functions.h"
#include "../../model/inc/model.h"

struct Image {
	uint *pixels;
	float *zbuffer;
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

struct DrawCallArgs {
	Image image{};
	ModelRef model{};
	float3 light_dir{};
	float3 camera_pos{};
	float3 look_dir{};
};

void render_init(int width, int height);

double main_cuda_launch(
        const DrawCallArgs &args,
        StopWatchInterface *timer
        );

#endif//COURSE_RENDERER_RENDER_CUH
