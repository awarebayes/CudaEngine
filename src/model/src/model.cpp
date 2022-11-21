//
// Created by dev on 7/9/22.
//
#include "../inc/model.h"
#include <glm/glm.hpp>
#include <helper_cuda.h>
#include <iostream>
#include <string>
#include <vector>

#include "../../util/tiny_objloader.h"

Sphere generateSphereBV(const std::vector<glm::vec3> &vertices)
{
	glm::vec3 minAABB = glm::vec3(std::numeric_limits<float>::max());
	glm::vec3 maxAABB = glm::vec3(0.001, 0.001, 0.001);

	for (const auto &i: vertices)
	{
		minAABB = glm::min(minAABB, i);
		maxAABB = glm::max(maxAABB, i);
	}

	return Sphere((maxAABB + minAABB) * 0.5f, glm::length(minAABB - maxAABB));
}


Model::Model(const tinyobj::ObjReader &reader, int index, const std::string &texture_search_path) {
	if (!reader.Warning().empty()) {
		std::cout << "TinyObjReader: " << reader.Warning();
	}

	auto &attrib = reader.GetAttrib();
	auto &shapes = reader.GetShapes();

	auto material_idx = shapes[index].mesh.material_ids[0];
	if (material_idx >= 0) {
		auto &material = reader.GetMaterials()[material_idx];
		if (!material.diffuse_texname.empty()) {
			std::string texture_filename = texture_search_path + material.diffuse_texname;
			texture = std::make_shared<Texture>(texture_filename);
		}
	}
	else
	{
		std::cout << "No material found for shape " << index << std::endl;
	}

	n_vertices = 0;
	for (auto i : shapes[index].mesh.indices)
		n_vertices = std::max(n_vertices, i.vertex_index);
	n_vertices += 1;

	n_faces = shapes[index].mesh.num_face_vertices.size();
	size_t uvertices = n_vertices;
	std::vector<glm::vec3> vertices_host{uvertices};
	std::vector<glm::vec3> normals_host{uvertices};
	std::vector<glm::vec2> textures_host{uvertices};
	std::vector<glm::ivec3> faces_host{};
	std::vector<glm::ivec3> textures_for_faces_host{size_t(n_faces)};

	bool has_textures = false;
	size_t index_offset = 0;
	int max_texture_index = 0;

	for (int face_idx = 0; face_idx < n_faces; face_idx ++)
	{
		unsigned char num_face_vertice =  shapes[index].mesh.num_face_vertices[face_idx];
		auto fv = size_t(num_face_vertice);
		assert(fv == 3);
		int indexes[3] = {0, 0, 0};

		// Loop over vertices in the face.
		for (size_t v = 0; v < fv; v++) {
			// access to vertex
			tinyobj::index_t idx = shapes[index].mesh.indices[index_offset + v];
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
				if (idx.texcoord_index >= textures_host.capacity())
					textures_host.resize(idx.texcoord_index + 1);
				textures_host[idx.texcoord_index] = glm::vec2{tx, ty};
				textures_for_faces_host[face_idx][v] =idx.texcoord_index;
				max_texture_index = std::max(max_texture_index, idx.texcoord_index) + 1;
				has_textures = true;
			}
		}

		faces_host.push_back({indexes[0], indexes[1], indexes[2]});

		index_offset += fv;
	}

	n_vertices = vertices_host.size();

	checkCudaErrors(cudaMalloc((void **) (&faces), sizeof(glm::ivec3) * n_faces));
	checkCudaErrors(cudaMalloc((void **) (&vertices), sizeof(glm::vec3) * n_vertices));
	checkCudaErrors(cudaMalloc((void **) (&normals), sizeof(glm::vec3) * n_vertices));

	if (has_textures) {
		checkCudaErrors(cudaMalloc((void **) (&textures), sizeof(glm::vec2) * max_texture_index));
		checkCudaErrors(cudaMalloc((void **) (&textures_for_face), sizeof(glm::ivec3) * n_faces));
		m_max_texture_index = max_texture_index;
	}

	checkCudaErrors(cudaMemcpy(faces, faces_host.data(), n_faces * sizeof(glm::ivec3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(vertices, vertices_host.data(), n_vertices * sizeof(glm::vec3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(normals, normals_host.data(), n_vertices * sizeof(glm::vec3), cudaMemcpyHostToDevice));

	if (has_textures)
	{
		checkCudaErrors(cudaMemcpy(textures, textures_host.data(), max_texture_index * sizeof(glm::vec2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(textures_for_face, textures_for_faces_host.data(), n_faces * sizeof(glm::ivec3), cudaMemcpyHostToDevice));
	}

	std::cerr << "# Model loaded with v# " << vertices_host.size() << " f# " << faces_host.size() << std::endl;

	bounding_volume = generateSphereBV(vertices_host);
}

Model Model::from_file(const std::string &filename, int index) {

	tinyobj::ObjReaderConfig reader_config;
	reader_config.mtl_search_path = "./";
	tinyobj::ObjReader reader;

	if (!reader.ParseFromFile(filename, reader_config)) {
		if (!reader.Error().empty()) {
			std::cerr << "TinyObjReader: " << reader.Error();
		}
		exit(1);
	}

	return {reader, index};
}

Model::~Model() {
	cudaFree(faces);
	cudaFree(normals);
	cudaFree(textures);
	cudaFree(vertices);
}
ModelRef Model::get_ref() {
	return ModelRef{texture->get_ref(), vertices, normals, textures, textures_for_face, faces, n_vertices, n_faces, m_max_texture_index, id, &bounding_volume, shader};
}
void Model::load_texture(const std::string &filename) {
	texture = std::make_shared<Texture>(filename);
}
