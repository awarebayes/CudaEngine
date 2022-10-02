//
// Created by dev on 7/25/22.
//

#include "../../model/inc/model.h"
#include "matrix.cuh"
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

	float3 pts[3]{};
	float3 normals[3]{};
	float2 textures[3]{};
	float3 light_dir{};

	__device__ explicit Shader(ModelRef &mod, float3 light_dir_, const glm::mat4 &projection_, const glm::mat4 &view_, const glm::mat4 &model_matrix_)
	    : model(mod), light_dir(light_dir_), projection(projection_), view(view_), model_matrix(model_matrix_) {};

	__device__ __forceinline__ float4 vertex(int iface, int nthvert)
	{
		auto face = model.faces[iface];
		int index = at(face, nthvert);
		float3 v = model.vertices[index];
		auto mv = glm::vec4(v.x, v.y, v.z, 1.0f);

		normals[nthvert] = model.normals[index];
		textures[nthvert] = model.textures[index];
		auto proj = projection * (view * (model_matrix * mv));
		proj.x = (proj.x + 1.0f) * 1920.0f / 2.0f;
		proj.y = (proj.y + 1.0f) * 1080.0f / 2.0f;
		pts[nthvert] = float3{proj.x, proj.y, proj.z};

		return float4{ pts[nthvert].x, pts[nthvert].y, pts[nthvert].z, 1.0f};
	}


	__device__ bool fragment(float3 bar, uint &output_color)
	{
		float3 N{};
		float2 T{};
		for (int i = 0; i < 3; i++)
		{
			N += normals[i] * at(bar, i);
			T += textures[i] * at(bar, i);
		}

		uchar3 color_u = model.texture.get_uv(T.x, T.y);
		float4 color_f = float4{float(color_u.x), float(color_u.y), float(color_u.z), 255.0f} / 255.0f;

		float4 colorf = color_f * max(dot(light_dir, N), 0.0f);
		colorf.w = 1.0f;
		output_color = rgbaFloatToInt(colorf);
		return false;                              // no, we do not discard this pixel
	}
};