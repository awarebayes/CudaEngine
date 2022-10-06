//
// Created by dev on 10/6/22.
//

#ifndef COURSE_RENDERER_BASE_SHADER_CUH
#define COURSE_RENDERER_BASE_SHADER_CUH

template <typename T>
struct BaseShader
{
	__device__ explicit BaseShader(ModelRef &mod, glm::vec3 light_dir_, const glm::mat4 &projection_, const glm::mat4 &view_, const glm::mat4 &model_matrix_, glm::vec2 screen_size_)
	: model(mod), light_dir(light_dir_), projection(projection_), view(view_), model_matrix(model_matrix_), screen_size(screen_size_) {};

	glm::mat4 projection;
	glm::mat4 view;
	glm::mat4 model_matrix;

	ModelRef &model;

	glm::vec3 pts[3]{};
	glm::vec3 normals[3]{};
	glm::vec2 textures[3]{};
	glm::vec3 light_dir{};
	glm::vec2 screen_size{};

	__device__ __forceinline__ float4 vertex(int iface, int nthvert)
	{
		static_cast<T*>(this)->vertex_impl(iface, nthvert);
	}

	__device__ bool fragment(glm::vec3 bar, uint &output_color)
	{
		static_cast<T*>(this)->fragment_impl(bar, output_color);
	}
};

#endif//COURSE_RENDERER_BASE_SHADER_CUH
