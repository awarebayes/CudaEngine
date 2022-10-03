//
// Created by dev on 10/3/22.
//

#include "../inc/bounding_volume.h"

bool Sphere::is_on_frustum(const Frustum &frustum, const glm::mat4 &model_matrix) const {
	const glm::vec3 scale = glm::vec3(glm::length(model_matrix[0]), glm::length(model_matrix[1]), glm::length(model_matrix[2]));
	const auto &center = glm::vec3(model_matrix * glm::vec4(this->center, 1.0f));
	const auto &max_scale = std::max(std::max(scale.x, scale.y), scale.z);
	const auto &radius = this->radius * max_scale * 0.5f;
	Sphere global_sphere = Sphere{ center, radius };
	auto result = (global_sphere.is_on_forward_plan(frustum.left) &&
	               global_sphere.is_on_forward_plan(frustum.right) &&
	               global_sphere.is_on_forward_plan(frustum.top) &&
	               global_sphere.is_on_forward_plan(frustum.bottom) &&
	               global_sphere.is_on_forward_plan(frustum.near) &&
	               global_sphere.is_on_forward_plan(frustum.far)
	);
	return result;
}
