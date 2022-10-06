//
// Created by dev on 10/6/22.
//

#ifndef COURSE_RENDERER_BASE_SHADER_CUH
#define COURSE_RENDERER_BASE_SHADER_CUH

#include "../render/misc/draw_caller_args.cuh"
template <typename T>
struct BaseShader
{
	__device__ explicit BaseShader(ModelRef &mod, glm::vec3 light_dir_, const glm::mat4 &projection_, const glm::mat4 &view_, const glm::mat4 &model_matrix_, glm::vec2 screen_size_, const DrawCallBaseArgs &base_args_)
	: model(mod), light_dir(light_dir_), projection(projection_), view(view_), model_matrix(model_matrix_), screen_size(screen_size_), base_args(base_args_) {};

	glm::mat4 projection;
	glm::mat4 view;
	glm::mat4 model_matrix;

	ModelRef &model;
	const DrawCallBaseArgs &base_args;

	glm::vec3 pts[3]{};
	glm::vec3 normals[3]{};
	glm::vec2 textures[3]{};
	glm::vec3 light_dir{};
	glm::vec2 screen_size{};

	__device__ __forceinline__ float4 vertex(int iface, int nthvert, bool load_tex)
	{
		static_cast<T*>(this)->vertex_impl(iface, nthvert, load_tex);
	}

	__device__ bool fragment(glm::vec3 bar, uint &output_color)
	{
		static_cast<T*>(this)->fragment_impl(bar, output_color);
	}
};

#endif//COURSE_RENDERER_BASE_SHADER_CUH
