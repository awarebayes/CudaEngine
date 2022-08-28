//
// Created by dev on 8/27/22.
//

#ifndef COURSE_RENDERER_CAMERA_H
#define COURSE_RENDERER_CAMERA_H

#include "../kernels/inc/matrix.cuh"
#include "../util/singleton.h"
#include <vector_types.h>

struct Camera
{
	float3 position{};
	float2 yaw_pitch{}; // yaw pitch roll



	void display_menu();
	[[nodiscard]] float3 get_look_direction() const;
	mat<4, 4> get_projection_matrix();
	mat<4, 4> get_view_matrix();
};

using CameraSingleton = SingletonCreator<Camera>;

#endif//COURSE_RENDERER_CAMERA_H
