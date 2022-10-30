//
// Created by dev on 10/20/22.
//

#include "../../../shader/all.h"
#include "../../misc/image.cuh"
#include "geometry_upsampler.h"

/*
__global__ void upsample_faces(ModelRef virtual_model, const ModelDrawCallArgs model_args, bool *disabled_original, bool *disabled_virtual, int *index_position) {
	int position = blockIdx.x * blockDim.x + threadIdx.x;
	auto &model = model_args.model;
	int max_pos = model.n_faces;
	if (position >= max_pos)
		return;

	bool is_disabled = disabled_original[position];
	if (!is_disabled)
		return;
	int my_position = atomicAdd(index_position, 1);

	if (my_position * 4 >= virtual_model.n_faces || my_position * 9 >= virtual_model.n_vertices)
	{
		printf("exceed model capacity! my_position: %d, virtual_model.n_faces: %d\n", my_position, virtual_model.n_faces);
		return;
	}

	disabled_virtual[my_position] = false;
	auto &face = model.faces[position];
	for (int i = 0; i < 3; i++) {
		int index = face[i];
		virtual_model.vertices[my_position * 3 + i] = model.vertices[index];
		virtual_model.normals[my_position * 3 + i] = model.normals[index];
	}
	virtual_model.faces[my_position] = {my_position * 3, my_position * 3 + 1, my_position * 3 + 2};
	virtual_model.textures_for_face[my_position] = model.textures_for_face[position];

	auto virtual_index = virtual_model.textures_for_face[my_position];
	auto original_index = model.textures_for_face[position];
	for (int i = 0; i < 3; i++) {
		virtual_model.textures[virtual_index[i]] = model.textures[original_index[i]];
	}
}
*/

__device__ void add_triangle(ModelRef &virtual_model, glm::ivec3 &face, glm::vec3 vertices[3], glm::vec3 normals[3], glm::vec2 textures[3], bool *disabled_virtual, int *index_position)
{
	int my_position = atomicAdd(index_position, 1);

	if (my_position * 4 >= virtual_model.n_faces || my_position * 9 >= virtual_model.n_vertices)
	{
		printf("exceed model capacity! my_position: %d, virtual_model.n_faces: %d\n", my_position, virtual_model.n_faces);
		return;
	}

	disabled_virtual[my_position] = false;
	for (int i = 0; i < 3; i++) {
		virtual_model.vertices[my_position * 3 + i] = vertices[i];
		virtual_model.normals[my_position * 3 + i] = normals[i];
		virtual_model.textures[my_position * 3 + i] = textures[i];
	}

	virtual_model.faces[my_position] = {my_position * 3, my_position * 3 + 1, my_position * 3 + 2};
	virtual_model.textures_for_face[my_position] = { my_position * 3, my_position * 3 + 1, my_position * 3 + 2 };
}

__device__ void upsample(ModelRef &virtual_model, glm::ivec3 &face, glm::vec3 vertices[3], glm::vec3 normals[3], glm::ivec3 &textures_for_face, glm::vec2 *textures, bool *disabled_virtual, int *index_position)
{

}

__global__ void upsample_faces(ModelRef virtual_model, const ModelDrawCallArgs model_args, bool *disabled_original, bool *disabled_virtual, int *index_position) {
	int position = blockIdx.x * blockDim.x + threadIdx.x;
	auto &model = model_args.model;
	int max_pos = model.n_faces;
	if (position >= max_pos)
		return;

	bool is_disabled = disabled_original[position];
	if (!is_disabled)
		return;

	glm::ivec3 face = model.faces[position];
	glm::vec3 vertices[3] = {
		model.vertices[face[0]],
		model.vertices[face[1]],
		model.vertices[face[2]],
	};

	glm::vec3 normals[3] = {
		model.normals[face[0]],
		model.normals[face[1]],
		model.normals[face[2]],
	};
	glm::vec2 textures[3] = {
		model.textures[model.textures_for_face[position][0]],
		model.textures[model.textures_for_face[position][1]],
		model.textures[model.textures_for_face[position][2]],
	};

	add_triangle(virtual_model, face, vertices, normals, textures, disabled_virtual, index_position);
}


void GeometryUpsampler::async_upsample_geometry(const ModelDrawCallArgs &model_args, bool *disabled_faces_for_original, bool *disabled_faces_for_virtual) {
	auto &model = model_args.model;
	auto n_grid = model.n_faces / 32 + 1;
	auto n_block = dim3(32);

	cudaMemsetAsync(position, 0, sizeof(int), stream);
	cudaMemsetAsync(disabled_faces_for_virtual, 1, sizeof(bool) * virtual_model.n_faces, stream);
	upsample_faces<<<n_grid, n_block, 0, stream>>>(virtual_model, model_args, disabled_faces_for_original, disabled_faces_for_virtual, position);
}

GeometryUpsampler::GeometryUpsampler(ModelRef &virtual_model_, cudaStream_t stream_)  : virtual_model(virtual_model_), stream(stream_)
{
	cudaMalloc(&position, sizeof(int));
}
GeometryUpsampler::~GeometryUpsampler() {
	cudaFree(position);
}
