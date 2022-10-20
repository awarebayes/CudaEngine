//
// Created by dev on 10/20/22.
//

#include "../../../shader/all.h"
#include "../../misc/image.cuh"
#include "geometry_upsampler.h"

__global__ void upsample_faces(ModelRef virtual_model, const ModelDrawCallArgs model_args, bool *face_mask, int *index_position) {
	int position = blockIdx.x * blockDim.x + threadIdx.x;
	auto &model = model_args.model;
	int max_pos = model.n_faces;
	if (position >= max_pos)
		return;

	bool is_disabled = face_mask[position];
	if (!is_disabled)
		return;
	auto my_position = atomicAdd(index_position, 1);
	if (my_position >= virtual_model.n_faces)
		return;
	auto &face = model.faces[position];
	auto &virtual_face = virtual_model.faces[my_position];
	virtual_face = face;
	auto points = model.vertices;
	for (int i = 0; i < 3; i++) {
		auto &virtual_point = virtual_model.vertices[my_position * 3 + i];
	 	virtual_point = points[face[i]];
	}
}

void GeometryUpsampler::async_upsample_geometry(const ModelDrawCallArgs &model_args, bool *disabled_faces) {
	auto &model = model_args.model;
	auto n_grid = model.n_faces / 32 + 1;
	auto n_block = dim3(32);

	cudaMemsetAsync(position, 0, sizeof(int), stream);
	upsample_faces<<<n_grid, n_block, 0, stream>>>(virtual_model, model_args, disabled_faces, position);
}
GeometryUpsampler::GeometryUpsampler(ModelRef virtual_model_, cudaStream_t stream_)  : virtual_model(virtual_model_), stream(stream_)
{
	cudaMalloc(&position, sizeof(int));
}
GeometryUpsampler::~GeometryUpsampler() {
	cudaFree(position);
}
