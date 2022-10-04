//
// Created by dev on 8/27/22.
//
#include "camera.h"
#include <cmath>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <imgui.h>

glm::vec3 Camera::get_look_direction() const {
	float yawr = yaw / 180.0f * M_PI;
	float pitchr = pitch / 180.0f * M_PI;
	glm::vec3 direction;
	direction.x = std::cos(yawr) * std::cos(pitchr);
	direction.y = std::sin(pitchr);
	direction.z = std::sin(yawr) * std::cos(pitchr);
	return direction;
}

void Camera::display_menu() {
	if (ImGui::CollapsingHeader("Camera Controls")) {
		ImGui::SliderFloat2("Camera XY", &position.x, -10, 10);
		ImGui::SliderFloat("Camera Z", &position.z, -100, 100);
		ImGui::SliderFloat("Look dir yaw", &yaw, -179.0, 179.0);
		ImGui::SliderFloat("Look dir pitch", &pitch, -179.0, 179.0);
		ImGui::SliderFloat("FOV", &fov, 1, 179);
		ImGui::SliderFloat("zFar", &zfar, 90, 1000);
	}
}
glm::vec2 Camera::get_screen_size() const {
	return  screen_size;
}

glm::mat4 Camera::get_view_matrix() const
{
	glm::vec3 look_dir = get_look_direction();
	return glm::lookAt(position, position + get_look_direction(), up);
}
glm::mat4 Camera::get_projection_matrix() const {
	return glm::perspective(glm::radians(fov), screen_size.x / screen_size.y, znear, zfar);
}
