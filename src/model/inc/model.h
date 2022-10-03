//
// Created by dev on 7/9/22.
//

#ifndef COURSE_RENDERER_MODEL_H
#define COURSE_RENDERER_MODEL_H

#include "bounding_volume.h"
#include "texture.cuh"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "vector_types.h"
#include <glm/glm.hpp>


struct ModelRef {
	glm::vec3 *vertices{};
	glm::vec3 *normals{};
	glm::vec2 *textures{};
	glm::ivec3 *faces{};

	TextureRef texture{};

	int n_vertices = 0;
	int n_faces = 0;

	Sphere *bounding_volume;
};

struct Model {
	glm::vec3 *vertices;
	glm::vec3 *normals{};
	glm::vec2 *textures{};
	glm::ivec3 *faces{};

	Texture texture{};

 	Sphere bounding_volume {{}, 0};

	int n_vertices = 0;
	int n_faces = 0;
	explicit Model(const std::string &filename);
	void load_texture(const std::string &filename);
	~Model();
	[[nodiscard]] ModelRef get_ref();
};

#endif//COURSE_RENDERER_MODEL_H
