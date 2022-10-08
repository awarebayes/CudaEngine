//
// Created by dev on 7/9/22.
//

#ifndef COURSE_RENDERER_MODEL_H
#define COURSE_RENDERER_MODEL_H

#include "../../shader/registered_shaders.h"
#include "../../util/tiny_objloader.h"
#include "bounding_volume.h"
#include "texture.cuh"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "vector_types.h"
#include <glm/glm.hpp>


struct ModelRef {
	TextureRef texture{};
	glm::vec3 *vertices{};
	glm::vec3 *normals{};
	glm::vec2 *textures{};
	glm::ivec3 *textures_for_face{};
	glm::ivec3 *faces{};
	int n_vertices = 0;
	int n_faces = 0;
	int id = 0;
	Sphere *bounding_volume{};
	RegisteredShaders shader = RegisteredShaders::Default;
};

struct Model {
	glm::vec3 *vertices{};
	glm::vec3 *normals{};
	glm::vec2 *textures{};
	glm::ivec3 *textures_for_face{};
	glm::ivec3 *faces{};

	RegisteredShaders shader;
	std::shared_ptr<Texture> texture{};
 	Sphere bounding_volume {{}, 0};

	int n_vertices = 0;
	int n_faces = 0;
	int id = 0;

	Model(const tinyobj::ObjReader &reader, int index);
	static Model from_file(const std::string &filename, int index=0);
	void load_texture(const std::string &filename);
	~Model();
	[[nodiscard]] ModelRef get_ref();
};

#endif//COURSE_RENDERER_MODEL_H
