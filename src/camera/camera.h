//
// Created by dev on 8/27/22.
//

#ifndef COURSE_RENDERER_CAMERA_H
#define COURSE_RENDERER_CAMERA_H

#include "../util/singleton.h"

struct Camera
{
	float3 position{};
	float3 ypr{}; // yaw pitch roll


	[[nodiscard]] float3 get_look_direction() const;

	void display_menu();
};

using CameraSingleton = SingletonCreator<Camera>;

#endif//COURSE_RENDERER_CAMERA_H
