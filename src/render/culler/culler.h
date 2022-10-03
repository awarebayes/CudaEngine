//
// Created by dev on 10/3/22.
//

#ifndef COURSE_RENDERER_CULLER_H
#define COURSE_RENDERER_CULLER_H
#include "../../camera/camera.h"
#include <glm/glm.hpp>
#include <vector>

struct AABB
{
	glm::vec3 min;
	glm::vec3 max;
};

struct Plan
{
	glm::vec3 normal = {0, 1.0f, 0};
	float d = 0.0f;
};

struct Frustum
{
	Plan left;
	Plan right;
	Plan top;
	Plan bottom;
	Plan near;
	Plan far;
};

Frustum from_camera(const Camera &cam, float aspect);

class Culler {
	void cull(std::vector<bool> &visible, const std::vector<AABB> &aabb, const glm::mat4 &view_matrix, const glm::mat4 &projection_matrix);
};


#endif//COURSE_RENDERER_CULLER_H
