//
// Created by dev on 8/27/22.
//

#ifndef COURSE_RENDERER_CAMERA_H
#define COURSE_RENDERER_CAMERA_H

#include "../kernels/inc/matrix.cuh"
#include "../util/singleton.h"
#include <vector_types.h>
#include <glm/glm.hpp>

struct Camera
{
	glm::vec3 position{};
	glm::vec3 up{0, 1, 0};
	float yaw{};
	float pitch{};

	void display_menu();
	[[nodiscard]] glm::vec3 get_look_direction() const;
	glm::mat4 get_view_matrix();
};

using CameraSingleton = SingletonCreator<Camera>;

#endif//COURSE_RENDERER_CAMERA_H
