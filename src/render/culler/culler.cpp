//
// Created by dev on 10/3/22.
//

#include "culler.h"
#include "../../kernels/inc/render.cuh"

Frustum frustum_from_camera(const Camera &cam) {
	Frustum frustum;
	float half_v_side = cam.zfar * std::tan(glm::radians(cam.fov) / 2.0f);
	float aspect = cam.screen_size.x / cam.screen_size.y;
	float half_h_side = half_v_side * aspect;

	auto front = glm::normalize(cam.get_look_direction());
	auto front_mul_far = front * cam.zfar;

	frustum.near = Plan{ cam.position + front * cam.znear, front };
	frustum.far = Plan{ cam.position + front_mul_far, -front };

	auto right_v = glm::normalize(glm::cross(front, cam.up));

	frustum.right = Plan{ cam.position, glm::cross(cam.up, front_mul_far + right_v * half_h_side)};
	frustum.left = Plan{ cam.position, glm::cross(front_mul_far - right_v * half_h_side, cam.up)};

	frustum.top = Plan{ cam.position, glm::cross(right_v, front_mul_far - cam.up * half_v_side)};
	frustum.bottom = Plan{ cam.position, glm::cross(front_mul_far + cam.up * half_v_side, right_v)};
	return frustum;
}

DrawCallArgs Culler::cull(const DrawCallArgs &args, const Camera &cam)
{
	DrawCallArgs visible_draw_calls;
	visible_draw_calls.base = args.base;
	auto frustum = frustum_from_camera(cam);
	for (const auto &i: args.models) {
		auto sphere = i.model.bounding_volume;
		auto transform = i.model_matrix;
		bool visible = sphere->is_on_frustum(frustum, transform);
		if (visible) {
			visible_draw_calls.models.push_back(i);
		}
	}
	return visible_draw_calls;
}
