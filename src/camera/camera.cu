//
// Created by dev on 8/27/22.
//
#include "camera.h"
#include <cmath>
#include <imgui.h>

#include "../kernels/inc/matrix.cuh"
#include "helper_math.h"

float3 Camera::get_look_direction() const {
	float yaw = yaw_pitch.x / 180.0f * M_PI;
	float pitch = yaw_pitch.y / 180.0f * M_PI;
	float3 direction;
	direction.x = std::cos(yaw) * std::cos(pitch);
	direction.y = std::sin(pitch);
	direction.z = std::sin(yaw) * std::cos(pitch);


	return direction;
}

void Camera::display_menu() {
	ImGui::Begin("Camera Controls");
	ImGui::SliderFloat2("Camera XY", &position.x, -10, 10);
	ImGui::SliderFloat("Camera Z", &position.z, -100, 100);
	ImGui::SliderFloat2("Look dir", &yaw_pitch.x, -180, 180);
	ImGui::End();
}

mat<4,4> Camera::get_projection_matrix()
{
	mat<4,4> Projection = identity_matrix<4>();
	Projection.at(3, 2) = -1.f / position.z;
	return Projection;
}

mat<4,4> Camera::get_view_matrix()
{
	return lookat(position, position + get_look_direction(), {0, 1, 0});
}
