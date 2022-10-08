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
	Water
};

static const std::unordered_map<RegisteredShaders, std::string_view> registered_shaders_string = {
	{RegisteredShaders::Default, "default"},
	{RegisteredShaders::Water, "water"}
};

static const std::unordered_map<std::string_view, RegisteredShaders> registered_shaders_enum = {
	{"default", RegisteredShaders::Default},
	{"water", RegisteredShaders::Water}
};



#endif//COURSE_RENDERER_REGISTERED_SHADERS_H
