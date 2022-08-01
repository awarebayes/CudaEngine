//
// Created by dev on 7/25/22.
//

#include "../../model/inc/model.h"
#include "shader.cuh"
#include "util.cuh"

#define GLM_FORCE_CUDA 1
#include <cuda.h>
#include <glm/glm.hpp>


struct Shader {
	glm::mat4 transform_M;   //  Projection*ModelView

	ModelRef &model;

	glm::vec3 pts[3]{};
	glm::vec3 normals[3]{};
	glm::vec2 textures[3]{};
	glm::vec3 light_dir{};

	__device__ explicit Shader(ModelRef &mod, glm::vec3 light_dir_) : model(mod), light_dir(light_dir_) {};

	__device__ __forceinline__ glm::vec4 vertex(int iface, int nthvert)
	{
		auto face = model.faces[iface];
		int index = at(face, nthvert);
		glm::vec3 v = model.vertices[index];

		normals[nthvert] = model.normals[index];
		textures[nthvert] = model.textures[index];

		pts[nthvert] = transform_M * glm::vec4(v.x, v.y, v.z, 1.0f);

		return glm::vec4{ pts[nthvert].x, pts[nthvert].y, pts[nthvert].z, 1.0f};
	}

	__device__ __forceinline__ bool fragment(glm::vec3 bar, uint &output_color)
	{
		glm::vec3 N{};
		glm::vec2 T{};
		for (int i = 0; i < 3; i++)
		{
			N += normals[i] * at(bar, i);
			T += textures[i] * at(bar, i);
		}

		uchar3 color_u = model.texture.get_uv(T.x, T.y);
		glm::vec4 color_f = glm::vec4{float(color_u.x), float(color_u.y), float(color_u.z), 255.0f} / 255.0f;

		// auto light_dir_ = m2v(dot(uniform_M,v2m(light_dir)));

		glm::vec4 colorf = color_f * glm::clamp(dot(light_dir, N), 0.0f, 1.0f);
		colorf.w = 1.0f;
		output_color = rgbaFloatToInt(colorf);
		return false;                              // no, we do not discard this pixel
	}
};