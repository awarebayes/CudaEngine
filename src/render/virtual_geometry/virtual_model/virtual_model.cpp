//
// Created by dev on 10/8/22.
//

#include "virtual_model.h"
#include "../../../util/const.h"
#include <helper_cuda.h>

ModelDrawCallArgs VirtualModel::to_args() {
	assert(scene_object_id.has_value());
	assert(false && "not implemented");
	return {
		.model = vmodel,
		.disabled_faces = disabled_faces,

	};
}

VirtualModel::VirtualModel() {
	vmodel.n_faces = 0;
	vmodel.n_vertices = 0;
}
VirtualModel::~VirtualModel() {
	free();
}

void VirtualModel::accept(ModelDrawCallArgs args, int n_bad_faces, int n_bad_vertices, int max_texture_index) {
	if (scene_object_id.has_value())
		release();

	scene_object_id = args.scene_object_id;

	if (n_bad_faces > vmodel.n_faces)
	{
		cudaFree(vmodel.faces);
		cudaFree(vmodel.textures_for_face);
		checkCudaErrors(cudaMalloc((void **) (&vmodel.faces), sizeof(glm::ivec3) * n_bad_faces));
		checkCudaErrors(cudaMalloc((void **) (&vmodel.textures_for_face), sizeof(glm::ivec3) * n_bad_faces));
	}
	if (n_bad_vertices)
	{
		checkCudaErrors(cudaMalloc((void **) (&vmodel.vertices), sizeof(glm::vec3) * n_bad_vertices));
		checkCudaErrors(cudaMalloc((void **) (&vmodel.normals), sizeof(glm::vec3) * n_bad_vertices));
	}
	if (max_texture_index > vmodel.n_vertices)
	{
		checkCudaErrors(cudaMalloc((void **) (&vmodel.textures), sizeof(glm::vec2) * max_texture_index));
	}
}
void VirtualModel::free() {
	cudaFree(vmodel.faces);
	cudaFree(vmodel.vertices);
	cudaFree(vmodel.normals);
	cudaFree(vmodel.textures);
	cudaFree(vmodel.textures_for_face);
}
void VirtualModel::release() {
	scene_object_id = std::nullopt;
}
int VirtualModel::get_model_id() {
	assert(scene_object_id.has_value());
	return vmodel.id;
}
