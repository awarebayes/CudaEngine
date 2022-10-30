//
// Created by dev on 10/30/22.
//

#ifndef COURSE_RENDERER_SHADER_VGEOM_CUH
#define COURSE_RENDERER_SHADER_VGEOM_CUH

struct ShaderVGeom : BaseShader<ShaderVGeom> {

	__device__ int hash(int x) {
		x = ((x >> 16) ^ x) * 0x45d9f3b;
		x = ((x >> 16) ^ x) * 0x45d9f3b;
		x = (x >> 16) ^ x;
		return x;
	}

	__device__ explicit ShaderVGeom(ModelRef &mod, glm::vec3 light_dir_, const glm::mat4 &projection_, const glm::mat4 &view_, const glm::mat4 &model_matrix_, glm::vec2 screen_size_, const DrawCallBaseArgs &args)
	    : BaseShader<ShaderVGeom>(mod, light_dir_, projection_, view_, model_matrix_, screen_size_, args) {};
	__device__ __forceinline__ float4 vertex_impl(int iface, int nthvert, bool load_tex)
	{
		auto face = model.faces[iface];
		int index = face[nthvert];
		glm::vec3 v = model.vertices[index];
		auto mv = glm::vec4(v.x, v.y, v.z, 1.0f);

		if (load_tex) {
			normals[nthvert] = model.normals[index];
			int texture = model.textures_for_face[iface][nthvert];
			textures[nthvert] = model.textures[texture];
		}

		auto proj = projection * (view * (model_matrix * mv));
		proj.w = abs(proj.w);
		proj.x = (proj.x + 1.0f) * screen_size.x / proj.w;
		proj.y = (proj.y + 1.0f) * screen_size.y / proj.w;
		proj.z = (proj.z + 1.0f) / proj.w;
		pts[nthvert] = glm::vec3{proj.x, proj.y, proj.z};
		position = (int)(proj.x + proj.y);
		return float4{ pts[nthvert].x, pts[nthvert].y, pts[nthvert].z, 1.0f};
	}

	__device__ bool fragment_impl(glm::vec3 bar, uint &output_color, float z_value)
	{
		int h = hash(this->position);
		int r = (h & 0xFF0000) >> 16;
		int g = (h & 0x00FF00) >> 8;
		int b = (h & 0x0000FF);
		output_color = rgbaFloatToInt({r / 255.0f, g / 255.0f, b / 255.0f, 1.0f});
		return false;
	}
};


#endif//COURSE_RENDERER_SHADER_VGEOM_CUH
