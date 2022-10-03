//
// Created by dev on 7/25/22.
//

#include "../../model/inc/model.h"
#include "shader.cuh"
#include "util.cuh"

#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_math.h>
#include <glm/glm.hpp>

struct Shader {
	glm::mat4 projection;
	glm::mat4 view;
	glm::mat4 model_matrix;

	ModelRef &model;

	glm::vec3 pts[3]{};
	glm::vec3 normals[3]{};
	glm::vec2 textures[3]{};
	glm::vec3 light_dir{};
	glm::vec2 screen_size{};

	__device__ explicit Shader(ModelRef &mod, glm::vec3 light_dir_, const glm::mat4 &projection_, const glm::mat4 &view_, const glm::mat4 &model_matrix_, glm::vec2 screen_size_)
	    : model(mod), light_dir(light_dir_), projection(projection_), view(view_), model_matrix(model_matrix_), screen_size(screen_size_) {};

	__device__ __forceinline__ float4 vertex(int iface, int nthvert)
	{
		auto face = model.faces[iface];
		int index = face[nthvert];
		glm::vec3 v = model.vertices[index];
		auto mv = glm::vec4(v.x, v.y, v.z, 1.0f);

		normals[nthvert] = model.normals[index];
		textures[nthvert] = model.textures[index];
		auto proj = projection * (view * (model_matrix * mv));
		proj.w = abs(proj.w);
		proj.x = (proj.x + 1.0f) / 2.0f * screen_size.x / proj.w;
		proj.y = (proj.y + 1.0f) / 2.0f * screen_size.y / proj.w;
		proj.z = (proj.z + 1.0f) / proj.w;
		pts[nthvert] = glm::vec3{proj.x, proj.y, proj.z};
		return float4{ pts[nthvert].x, pts[nthvert].y, pts[nthvert].z, 1.0f};
	}


	__device__ bool fragment(glm::vec3 bar, uint &output_color)
	{
		glm::vec3 N{};
		glm::vec2 T{};
		for (int i = 0; i < 3; i++)
		{
			N += normals[i] * bar[i];
			T += textures[i] * bar[i];
		}

		uchar3 color_u = model.texture.get_uv(T.x, T.y);
		float4 color_f = float4{float(color_u.x), float(color_u.y), float(color_u.z), 255.0f} / 255.0f;

		float4 colorf = color_f * max(dot(light_dir, N), 0.0f);
		colorf.w = 1.0f;
		output_color = rgbaFloatToInt(colorf);
		return false;                              // no, we do not discard this pixel
	}
};