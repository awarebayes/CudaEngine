//
// Created by dev on 8/27/22.
//
#include "camera.h"

float3 Camera::get_look_direction() const {
	float yaw = ypr.x;
	float pitch = ypr.y;
	float roll = ypr.z;

	float x = -std::cos(yaw) * std::sin(pitch)*sin(roll)-std::sin(yaw)*std::cos(roll);
	float y = -std::sin(yaw) * std::sin(pitch)*sin(roll)+std::cos(yaw)*std::cos(roll);
	float z =  std::cos(pitch) * std::sin(roll);
	return {x, y, z};
}
