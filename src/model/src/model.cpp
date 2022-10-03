//
// Created by dev on 7/9/22.
//
#include "../inc/model.h"
#include <fstream>
#include <glm/glm.hpp>
#include <helper_cuda.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../util/tiny_objloader.h"

Model::Model(const std::string &filename) : vertices(), faces() {

	tinyobj::ObjReaderConfig reader_config;
	reader_config.mtl_search_path = "./";
	tinyobj::ObjReader reader;

	if (!reader.ParseFromFile(filename, reader_config)) {
		if (!reader.Error().empty()) {
			std::cerr << "TinyObjReader: " << reader.Error();
		}
		exit(1);
	}

	if (!reader.Warning().empty()) {
		std::cout << "TinyObjReader: " << reader.Warning();
	}

	auto &attrib = reader.GetAttrib();
	auto &shapes = reader.GetShapes();
	assert(shapes.size() == 1);

	n_vertices = 0;
	for (auto i : shapes[0].mesh.indices)
		n_vertices = std::max(n_vertices, i.vertex_index);
	n_vertices += 1;

	size_t uvertices = n_vertices;
	std::vector<glm::vec3> vertices_host{uvertices};
	std::vector<glm::vec3> normals_host{uvertices};
	std::vector<glm::vec2> textures_host{uvertices};
	std::vector<glm::ivec3> faces_host{};

	bool has_textures = false;

	size_t index_offset = 0;
	for (unsigned char num_face_vertice : shapes[0].mesh.num_face_vertices) {
		auto fv = size_t(num_face_vertice);
		assert(fv == 3);
		int indexes[3] = {0, 0, 0};

		// Loop over vertices in the face.
		for (size_t v = 0; v < fv; v++) {
			// access to vertex
			tinyobj::index_t idx = shapes[0].mesh.indices[index_offset + v];
			indexes[v] = idx.vertex_index;
			tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
			tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
			tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

			vertices_host[idx.vertex_index] = glm::vec3{vx, vy, vz};

			// Check if `normal_index` is zero or positive. negative = no normal data
			if (idx.normal_index >= 0) {
				tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
				tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
				tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
				normals_host[idx.vertex_index] = glm::vec3{nx, ny, nz};
			}

			// Check if `texcoord_index` is zero or positive. negative = no texcoord data
			if (idx.texcoord_index >= 0) {
				tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
				tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
				textures_host[idx.vertex_index] = glm::vec2{tx, ty};
				has_textures = true;
			}
		}

		faces_host.push_back({indexes[0], indexes[1], indexes[2]});

		index_offset += fv;
	}

	n_vertices = vertices_host.size();
	n_faces = faces_host.size();

	checkCudaErrors(cudaMalloc((void **) (&faces), sizeof(glm::ivec3) * n_faces));
	checkCudaErrors(cudaMalloc((void **) (&vertices), sizeof(glm::vec3) * n_vertices));
	checkCudaErrors(cudaMalloc((void **) (&normals), sizeof(glm::vec3) * n_vertices));

	if (has_textures)
		checkCudaErrors(cudaMalloc((void **) (&textures), sizeof(glm::vec3) * n_vertices));

	checkCudaErrors(cudaMemcpy(faces, faces_host.data(), n_faces * sizeof(glm::ivec3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(vertices, vertices_host.data(), n_vertices * sizeof(glm::vec3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(normals, normals_host.data(), n_vertices * sizeof(glm::vec3), cudaMemcpyHostToDevice));

	if (has_textures)
		checkCudaErrors(cudaMemcpy(textures, textures_host.data(), n_vertices * sizeof(glm::vec2), cudaMemcpyHostToDevice));

	std::cerr << "# Model loaded with v# " << vertices_host.size() << " f# " << faces_host.size() << std::endl;
}

Model::~Model() {
	checkCudaErrors(cudaFree(faces));
	checkCudaErrors(cudaFree(vertices));
	checkCudaErrors(cudaFree(normals));
	checkCudaErrors(cudaFree(textures));
}
ModelRef Model::get_ref() const {
	return ModelRef{vertices, normals, textures, faces, texture.get_ref(), n_vertices, n_faces};
}
void Model::load_texture(const std::string &filename) {
	texture = Texture(filename);
}
