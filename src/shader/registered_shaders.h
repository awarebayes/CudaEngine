//
// Created by dev on 10/6/22.
//

#ifndef COURSE_RENDERER_REGISTERED_SHADERS_H
#define COURSE_RENDERER_REGISTERED_SHADERS_H

#include <string_view>
#include <unordered_map>

enum class RegisteredShaders
{
	Default,
	Water,
	VGeom
};

static const std::unordered_map<RegisteredShaders, std::string_view> registered_shaders_string = {
	{RegisteredShaders::Default, "default"},
	{RegisteredShaders::Water, "water"},
	{RegisteredShaders::VGeom, "vgeom"}
};

static const std::unordered_map<std::string_view, RegisteredShaders> registered_shaders_enum = {
	{"default", RegisteredShaders::Default},
	{"water", RegisteredShaders::Water},
	{"vgeom", RegisteredShaders::VGeom}
};



#endif//COURSE_RENDERER_REGISTERED_SHADERS_H
