//
// Created by dev on 8/27/22.
//
#include "camera.h"
#include <imgui.h>

float3 Camera::get_look_direction() const {
	float yaw = ypr.x / 180.0f * M_PI;
	float pitch = ypr.y / 180.0f * M_PI;
	float roll = ypr.z / 180.0f * M_PI;
	float3 direction;
	direction.x = cos(yaw) * cos(pitch);
	direction.y = sin(pitch);
	direction.z = sin(yaw) * cos(pitch);


	return direction;
}
void Camera::display_menu() {
	ImGui::Begin("Camera Controls");
	ImGui::SliderFloat3("Camera XYZ", &position.x, -10, 10);
	ImGui::SliderFloat3("Look dir", &ypr.x, -180, 180);
	ImGui::End();
}
