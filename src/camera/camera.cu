//
// Created by dev on 8/27/22.
//
#include "camera.h"
#include <cmath>
#include <glm/ext/matrix_transform.hpp>
#include <imgui.h>

#include "../kernels/inc/matrix.cuh"
#include "helper_math.h"

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
	ImGui::Begin("Camera Controls");
	ImGui::SliderFloat2("Camera XY", &position.x, -10, 10);
	ImGui::SliderFloat("Camera Z", &position.z, -100, 100);
	ImGui::SliderFloat("Look dir yaw", &yaw, -180, 180);
	ImGui::SliderFloat("Look dir pitch", &pitch, -180, 180);
	ImGui::End();
}

glm::mat4 Camera::get_view_matrix()
{
	return glm::lookAt(position, position + get_look_direction(), up);
}
