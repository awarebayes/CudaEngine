//
// Created by dev on 10/6/22.
//
#include "../../../shader/all.h"
#include "../../../util/const.h"
#include "mesh_analyzer.h"

template <typename ShaderType>
__global__ void analyze_faces(DrawCallBaseArgs args, ModelDrawCallArgs model_args, int threshold, float *surface_areas, int n_faces, bool *has_bad_faces) {
	int position = blockIdx.x * blockDim.x + threadIdx.x;
	auto &model = model_args.model;
	int max_pos = max(model.n_faces, n_faces);
	if (position >= max_pos)
		return;

	auto sh = BaseShader<ShaderType>(model, args.light_dir, args.projection, args.view, model_args.model_matrix, args.screen_size, args);

	for (int i = 0; i < 3; i++)
		sh.vertex(position, i, false);


	auto &pts = sh.pts;

	if (pts[0].y==pts[1].y && pts[0].y==pts[2].y) return;

	*has_bad_faces = false;
}

void MeshAnalyzer::async_analyze_mesh(const DrawCallArgs &args, int model_index)
{
	auto &model_args = args.models[model_index];
	auto &model = model_args.model;
	auto n_grid = std::min(model.n_faces, VIRTUAL_GEOMETRY_FACES) / 32 + 1;
	auto n_block = dim3(32);

	switch (model.shader)
	{
		case RegisteredShaders::Default:
			analyze_faces<ShaderDefault><<<n_grid, n_block, 0, stream>>>(args.base, model_args, threshold, surface_areas, capacity, has_bad_faces);
			break;
		case RegisteredShaders::Water:
			analyze_faces<ShaderWater><<<n_grid, n_block, 0, stream>>>(args.base, model_args, threshold, surface_areas, capacity, has_bad_faces);
			break;
	}
}
MeshAnalyzer::MeshAnalyzer(int capacity_, int threshold_) : capacity(capacity_), threshold(threshold_), Synchronizable() {
	cudaMalloc(&surface_areas, sizeof(float) * capacity);
}
MeshAnalyzer::~MeshAnalyzer() {
	cudaFree(surface_areas);
}
