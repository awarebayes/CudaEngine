//
// Created by dev on 10/20/22.
//

#include "../../../shader/all.h"
#include "../../misc/image.cuh"
#include "geometry_upsampler.h"

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

	disabled_virtual[my_position] = false;
	if (my_position >= virtual_model.n_faces)
	{
		printf("exceed model capacity! my_position: %d, virtual_model.n_faces: %d\n", my_position, virtual_model.n_faces);
		return;
	}

	auto &face = model.faces[position];
	for (int i = 0; i < 3; i++) {
		int index = face[i];
		virtual_model.vertices[my_position * 3 + i] = model.vertices[index];
		virtual_model.normals[my_position * 3 + i] = model.normals[index];
	}
	virtual_model.faces[my_position] = {my_position * 3, my_position * 3 + 1, my_position * 3 + 2};
	virtual_model.textures_for_face[my_position] = model.textures_for_face[position];
}

void GeometryUpsampler::async_upsample_geometry(const ModelDrawCallArgs &model_args, bool *disabled_faces_for_original, bool *disabled_faces_for_virtual) {
	auto &model = model_args.model;
	auto n_grid = model.n_faces / 32 + 1;
	auto n_block = dim3(32);

	cudaMemsetAsync(position, 0, sizeof(int), stream);
	cudaMemsetAsync(disabled_faces_for_virtual, 0, sizeof(bool) * virtual_model.n_faces, stream);
	upsample_faces<<<n_grid, n_block, 0, stream>>>(virtual_model, model_args, disabled_faces_for_original, disabled_faces_for_virtual, position);

	//char *temp = (char *)malloc(100000);
	//cudaMemcpyAsync(temp, disabled_faces_for_virtual, sizeof(bool) * virtual_model.n_faces, cudaMemcpyDeviceToHost, stream);
	//free(temp);
}

GeometryUpsampler::GeometryUpsampler(ModelRef &virtual_model_, cudaStream_t stream_)  : virtual_model(virtual_model_), stream(stream_)
{
	cudaMalloc(&position, sizeof(int));
}
GeometryUpsampler::~GeometryUpsampler() {
	cudaFree(position);
}
