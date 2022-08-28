//
// Created by dev on 8/27/22.
//

#ifndef COURSE_RENDERER_CAMERA_H
#define COURSE_RENDERER_CAMERA_H

struct Camera
{
	float3 position{};
	float3 ypr{}; // yaw pitch roll

	[[nodiscard]] float3 get_look_direction() const;
};

#endif//COURSE_RENDERER_CAMERA_H
