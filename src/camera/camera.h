//
// Created by dev on 8/27/22.
//

#ifndef COURSE_RENDERER_CAMERA_H
#define COURSE_RENDERER_CAMERA_H

#include "../util/singleton.h"
#include <vector_types.h>
#include <glm/glm.hpp>

struct Camera
{
	glm::vec3 position{};
	glm::vec3 up{0, 1, 0};
	float yaw{};
	float pitch{};
	float fov{45};
	float znear{0.1f};
	float zfar{100.0f};

	glm::vec2 screen_size{1920, 1080};

	void display_menu();
	[[nodiscard]] glm::vec3 get_look_direction() const;
	glm::mat4 get_view_matrix() const;
	glm::mat4 get_projection_matrix() const;
	glm::vec2 get_screen_size() const;
};

using CameraSingleton = SingletonCreator<Camera>;

#endif//COURSE_RENDERER_CAMERA_H
