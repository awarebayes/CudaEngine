//
// Created by dev on 7/8/22.
//

#ifndef COURSE_RENDERER_RENDER_CUH
#define COURSE_RENDERER_RENDER_CUH

#include "../../../Common/helper_functions.h"
#include "../../camera/camera.h"
#include "../../model/inc/model.h"
#include <glm/glm.hpp>
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
	glm::vec3 light_dir{};
	glm::vec3 camera_pos{};
	glm::vec3 look_at{};
	glm::mat4 view{};
	glm::mat4 projection{};
	glm::vec2 screen_size{1920, 1080};
	Camera *camera_ptr = nullptr;
};

struct ModelArgs
{
	glm::mat4 model_matrix{};
	ModelRef model;
};

struct DrawCallArgs {
	std::vector<ModelArgs> models{};
	DrawCallBaseArgs base{};
};

struct StoredModel
{
	glm::vec3 position{};
	ModelRef model{};
	ModelArgs to_args();
};


template <typename Tp>
__device__ __forceinline__ glm::vec3 barycentric(glm::vec3 *pts, Tp P) {
	auto a = glm::vec3{float(pts[2].x-pts[0].x), float(pts[1].x-pts[0].x), float(pts[0].x-P.x)};
	auto b = glm::vec3{float(pts[2].y-pts[0].y), float(pts[1].y-pts[0].y), float(pts[0].y-P.y)};
	auto u = glm::cross(a, b);
	float flag = abs(u.z) < 1;
	return glm::vec3{
			-1.0f * flag + (1.0f - flag) * (1.f-(u.x+u.y)/u.z),
			1.0f * flag + (1.0f - flag) * (u.y/u.z),
			1.0f * flag + (1.0f - flag) * (u.x/u.z)
	};
}

#endif//COURSE_RENDERER_RENDER_CUH
