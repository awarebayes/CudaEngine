//
// Created by dev on 7/25/22.
//


#include "../model/inc/model.h"
#include "../render/misc/util.cuh"
#include "base_shader.cuh"
#include <glm/glm.hpp>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_math.h>

struct ShaderDefault : BaseShader<ShaderDefault> {

	__device__ explicit ShaderDefault(ModelRef &mod, glm::vec3 light_dir_, const glm::mat4 &projection_, const glm::mat4 &view_, const glm::mat4 &model_matrix_, glm::vec2 screen_size_)
	        : BaseShader<ShaderDefault>(mod, light_dir_, projection_, view_, model_matrix_, screen_size_) {};
	__device__ __forceinline__ float4 vertex_impl(int iface, int nthvert)
	{
		auto face = model.faces[iface];
		int index = face[nthvert];
		glm::vec3 v = model.vertices[index];
		auto mv = glm::vec4(v.x, v.y, v.z, 1.0f);

		normals[nthvert] = model.normals[index];
		textures[nthvert] = model.textures[model.textures_for_face[iface][nthvert]];

		auto proj = projection * (view * (model_matrix * mv));
		proj.w = abs(proj.w);
		proj.x = (proj.x + 1.0f) * screen_size.x / proj.w;
		proj.y = (proj.y + 1.0f) * screen_size.y / proj.w;
		proj.z = (proj.z + 1.0f) / proj.w;
		pts[nthvert] = glm::vec3{proj.x, proj.y, proj.z};
		return float4{ pts[nthvert].x, pts[nthvert].y, pts[nthvert].z, 1.0f};
	}


	__device__ bool fragment_impl(glm::vec3 bar, uint &output_color)
	{
		glm::vec3 N = glm::normalize(normals[0] * bar.x + normals[1] * bar.y + normals[2] * bar.z);
		glm::vec2 uv = textures[0] * bar.x + textures[1] * bar.y + textures[2] * bar.z;

		uchar3 color_u = model.texture.get_uv(uv.x, uv.y);
		float4 color_f = float4{float(color_u.x), float(color_u.y), float(color_u.z), 255.0f} / 255.0f;

		color_f = color_f * max(dot(light_dir, N), 0.0f);
		color_f.w = 1.0f;
		output_color = rgbaFloatToInt(color_f);
		return false;
	}
};