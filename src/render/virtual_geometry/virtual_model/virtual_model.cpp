//
// Created by dev on 10/8/22.
//

#include "virtual_model.h"
#include "../../../util/const.h"
#include <helper_cuda.h>

ModelDrawCallArgs VirtualModel::to_args() {
	assert(holding_id.has_value());
	return {};
}
VirtualModel::VirtualModel() {
	int n_vertices = VIRTUAL_GEOMETRY_FACES;
	int n_faces = VIRTUAL_GEOMETRY_VERTICES;
	int max_texture_index = VIRTUAL_GEOMETRY_FACES;

	checkCudaErrors(cudaMalloc((void **) (&vmodel.faces), sizeof(glm::ivec3) * n_faces));
	checkCudaErrors(cudaMalloc((void **) (&vmodel.vertices), sizeof(glm::vec3) * n_vertices));
	checkCudaErrors(cudaMalloc((void **) (&vmodel.normals), sizeof(glm::vec3) * n_vertices));

	checkCudaErrors(cudaMalloc((void **) (&vmodel.textures), sizeof(glm::vec3) * max_texture_index));
	checkCudaErrors(cudaMalloc((void **) (&vmodel.textures_for_face), sizeof(glm::ivec3) * n_faces));
}
VirtualModel::~VirtualModel() {

}
