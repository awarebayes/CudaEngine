//
// Created by dev on 10/6/22.
//
#include "../../../shader/all.h"
#include "../../../util/const.h"
#include "mesh_analyzer.h"

__device__ int get_subdivision_count(float area, int threshold) {
	float unit = area;
	int count = 1;
	while (unit > threshold) {
		unit /= 4;
		count *= 4;
	}
	return count;
}

template <typename ShaderType>
__global__ void analyze_faces(DrawCallBaseArgs args, ModelDrawCallArgs model_args, const Image image, int threshold, bool *face_mask, int n_faces, int *new_virtual_faces) {
	int position = blockIdx.x * blockDim.x + threadIdx.x;
	auto &model = model_args.model;
	int max_pos = model.n_faces;
	if (position >= max_pos)
		return;

	auto sh = BaseShader<ShaderType>(model, args.light_dir, args.projection, args.view, model_args.model_matrix, args.screen_size, args);
	for (int i = 0; i < 3; i++)
			sh.vertex(position, i, false);

	auto &pts = sh.pts;
	if (pts[0].y==pts[1].y && pts[0].y==pts[2].y) return;

	glm::vec2 bboxmin{float(image.width-1),  float(image.height-1)};
	glm::vec2 bboxmax{0.01, 0.01};
	glm::vec2 clamp{float(image.width-1), float(image.height-1)};
	for (auto &pt : pts) {
		bboxmin.x = max(0.0f, min(bboxmin.x, pt.x));
		bboxmin.y = max(0.0f, min(bboxmin.y, pt.y));

		bboxmax.x = min(clamp.x, max(bboxmax.x, pt.x));
		bboxmax.y = min(clamp.y, max(bboxmax.y, pt.y));
	}


	float area = (bboxmax.x - bboxmin.x) * (bboxmax.y - bboxmin.y);
	if (area > threshold) {
		auto count = get_subdivision_count(area, threshold);
		atomicAdd(new_virtual_faces, count);
		face_mask[position] = true;
	}
}

void MeshAnalyzer::async_analyze_mesh(const DrawCallArgs &args, const Image &image, int model_index)
{
	auto &model_args = args.models[model_index];
	auto &model = model_args.model;
	auto n_grid = model.n_faces / 32 + 1;
	auto n_block = dim3(32);

	if (model.n_faces > capacity) {
		capacity = model.n_faces;
		cudaFreeAsync(face_mask, stream);
		cudaMallocAsync(&face_mask, sizeof(float) * capacity, stream);
	}

	cudaMemsetAsync(face_mask, 0, sizeof(bool) * model.n_faces, stream);
	switch (model.shader)
	{
		case RegisteredShaders::Default:
			analyze_faces<ShaderDefault><<<n_grid, n_block, 0, stream>>>(args.base, model_args, image, area_threshold, face_mask, capacity, new_vfaces_count);
			break;
		case RegisteredShaders::Water:
			analyze_faces<ShaderWater><<<n_grid, n_block, 0, stream>>>(args.base, model_args, image, area_threshold, face_mask, capacity, new_vfaces_count);
			break;
		case RegisteredShaders::VGeom:
			analyze_faces<ShaderVGeom><<<n_grid, n_block, 0, stream>>>(args.base, model_args, image, area_threshold, face_mask, capacity, new_vfaces_count);
			break;
	}
}
MeshAnalyzer::MeshAnalyzer(int capacity_, int &threshold_) : capacity(capacity_), area_threshold(threshold_), Synchronizable() {
	cudaMalloc(&face_mask, sizeof(bool) * capacity);
}
MeshAnalyzer::~MeshAnalyzer() {
	cudaFree(&face_mask);
}
