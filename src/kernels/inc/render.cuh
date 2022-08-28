//
// Created by dev on 7/8/22.
//

#ifndef COURSE_RENDERER_RENDER_CUH
#define COURSE_RENDERER_RENDER_CUH

#include "../../../Common/helper_functions.h"
#include "../../model/inc/model.h"
#include "matrix.cuh"
#include <helper_math.h>

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

struct DrawCallBaseArgs
{
	mat<4, 4> model_matrix{};
	float3 light_dir{};
	float3 camera_pos{};
	float3 look_at{};
};

struct DrawCallArgs {
	std::vector<ModelRef> models{};
	DrawCallBaseArgs base{};
};

void render_init(int width, int height);

void update_device_parameters(const DrawCallArgs &args);

template <typename Tp>
__device__ __forceinline__ float3 barycentric(float3 *pts, Tp P) {
	auto a = float3{float(pts[2].x-pts[0].x), float(pts[1].x-pts[0].x), float(pts[0].x-P.x)};
	auto b = float3{float(pts[2].y-pts[0].y), float(pts[1].y-pts[0].y), float(pts[0].y-P.y)};
	auto u = cross(a, b);
	float flag = abs(u.z) < 1;
	return float3{
			-1.0f * flag + (1.0f - flag) * (1.f-(u.x+u.y)/u.z),
			1.0f * flag + (1.0f - flag) * (u.y/u.z),
			1.0f * flag + (1.0f - flag) * (u.x/u.z)
	};
}

#endif//COURSE_RENDERER_RENDER_CUH
