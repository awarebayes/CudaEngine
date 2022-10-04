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
__device__ __forceinline__ glm::vec3 barycentric(glm::vec3 a, glm::vec3 b, glm::vec3 c, Tp P) {

	auto m = glm::vec3{float(c.x-a.x), float(b.x-a.x), float(a.x-P.x)};
	auto n = glm::vec3{float(c.y-a.y), float(b.y-a.y), float(a.y-P.y)};
	auto u = glm::cross(n, m);
	if (abs(u.z) < 1)
		return glm::vec3(-1, 1, 1);
	return glm::vec3{1.f-(u.x+u.y)/u.z, u.y/u.z, u.x/u.z};
}

#endif//COURSE_RENDERER_RENDER_CUH
