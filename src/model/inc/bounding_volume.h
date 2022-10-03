//
// Created by dev on 10/3/22.
//

#ifndef COURSE_RENDERER_BOUNDING_VOLUME_H
#define COURSE_RENDERER_BOUNDING_VOLUME_H

#include <glm/glm.hpp>

struct Plan
{
	glm::vec3 normal = {0, 1.0f, 0};
	float d = 0.0f;

	Plan(glm::vec3 point, glm::vec3 normal_) : normal(normal_), d(glm::dot(normal, point)) {};
	Plan() = default;

	float getSignedDistanceToPlan(const glm::vec3& point) const
	{
		return glm::dot(normal, point) - d;
	}
};

struct Frustum
{
	Plan left;
	Plan right;
	Plan top;
	Plan bottom;
	Plan near;
	Plan far;

	Frustum() = default;
};

class Volume
{
public:
	[[nodiscard]] virtual bool is_on_frustum(const Frustum &frustum, const glm::mat4 &transform) const = 0;
};

class Sphere : public Volume {
protected:
	glm::vec3 center;
	float radius;
public:
	Sphere(glm::vec3 center_, float radius_) : center(center_), radius(radius_) {};
	[[nodiscard]] bool is_on_frustum(const Frustum &frustum, const glm::mat4 &model_matrix) const override;
	bool is_on_forward_plan(const Plan& plan) const
	{
		auto dtp = plan.getSignedDistanceToPlan(center);
		return dtp > -radius;
	}
};

#endif//COURSE_RENDERER_BOUNDING_VOLUME_H
