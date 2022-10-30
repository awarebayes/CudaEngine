//
// Created by dev on 10/8/22.
//

#include "virtual_model.h"
#include "../../../util/const.h"
#include <helper_cuda.h>

ModelDrawCallArgs VirtualModel::get_virtual_updated() {
	assert(scene_object_id.has_value());
	await();

	return {
	        .model = vmodel,
	        .disabled_faces = disabled_faces_for_virtual,
	};
}

VirtualModel::VirtualModel() {
	vmodel.n_faces = 0;
	vmodel.n_vertices = 0;
	vmodel.is_virtual = true;
	geometry_upsampler = std::make_unique<GeometryUpsampler>(vmodel, stream);
}

VirtualModel::~VirtualModel() {
	free();
}

void VirtualModel::accept(ModelDrawCallArgs args, bool *disabled_faces_to_copy) {
	assert(!scene_object_id.has_value());

	scene_object_id = args.scene_object_id;

	int max_texture_index = args.model.max_texture_index;

	auto multiplier = 9;

	if (args.model.n_faces * multiplier > n_allocated_faces)
	{
		n_allocated_faces = args.model.n_faces * multiplier;
		cudaFreeAsync(vmodel.faces, stream);
		cudaFreeAsync(vmodel.textures_for_face, stream);
		checkCudaErrors(cudaMallocAsync((void **) (&vmodel.faces), sizeof(glm::ivec3) * n_allocated_faces, stream));
		checkCudaErrors(cudaMallocAsync((void **) (&vmodel.textures_for_face), sizeof(glm::ivec3) * n_allocated_faces, stream));
	}
	if (args.model.n_vertices * multiplier > n_allocated_vertices)
	{
		n_allocated_vertices = args.model.n_vertices;
		checkCudaErrors(cudaMallocAsync((void **) (&vmodel.vertices), sizeof(glm::vec3) * n_allocated_vertices, stream));
		checkCudaErrors(cudaMallocAsync((void **) (&vmodel.normals), sizeof(glm::vec3) *  n_allocated_vertices, stream));
	}
	if (max_texture_index > vmodel.max_texture_index)
	{
		checkCudaErrors(cudaMallocAsync((void **) (&vmodel.textures), sizeof(glm::vec2) * max_texture_index, stream));
		vmodel.max_texture_index = max_texture_index;
	}
	if (args.model.n_faces > m_allocated_disabled_faces)
	{
		checkCudaErrors(cudaMalloc((void **) (&disabled_faces_for_original), sizeof(glm::ivec3) * args.model.n_faces));
		checkCudaErrors(cudaMalloc((void **) (&disabled_faces_for_virtual), sizeof(glm::ivec3) * args.model.n_faces));
		m_allocated_disabled_faces = args.model.n_faces;
	}

	vmodel.n_faces = n_allocated_faces;
	vmodel.n_vertices = n_allocated_vertices;
	vmodel.texture = args.model.texture;
	vmodel.shader = args.model.shader;
	vmodel.bounding_volume = args.model.bounding_volume;

	// cudaMemcpyAsync(vmodel.textures, args.model.textures, sizeof(glm::vec2) * max_texture_index, cudaMemcpyDeviceToDevice, stream);

	update_virtual_model(args, disabled_faces_to_copy);
}

void VirtualModel::free() {
	cudaFree(vmodel.faces);
	cudaFree(vmodel.vertices);
	cudaFree(vmodel.normals);
	cudaFree(vmodel.textures);
	cudaFree(vmodel.textures_for_face);
	cudaFree(disabled_faces_for_original);
	cudaFree(disabled_faces_for_virtual);
}

void VirtualModel::release() {
	assert(scene_object_id.has_value());
	scene_object_id = std::nullopt;
	clear();
	last_updated = std::chrono::system_clock::now();
}

int VirtualModel::get_model_id() {
	assert(scene_object_id.has_value());
	return vmodel.id;
}

void VirtualModel::update_virtual_model(ModelDrawCallArgs original_model, bool *disabled_faces_to_copy)
{
	cudaMemcpyAsync(disabled_faces_for_original, disabled_faces_to_copy, sizeof(bool) * m_allocated_disabled_faces, cudaMemcpyDeviceToDevice);
	geometry_upsampler->async_upsample_geometry(original_model, disabled_faces_for_original, disabled_faces_for_virtual);
	last_updated = std::chrono::system_clock::now();
}

void VirtualModel::update(ModelDrawCallArgs model, bool *disabled_faces_to_copy) {
	using namespace std::chrono_literals;
	assert(scene_object_id.has_value());
	assert(scene_object_id.value() == model.scene_object_id);
	auto since_update = (std::chrono::system_clock::now() - last_updated) / 1ms;
	if (since_update < UPDATE_VIRTUAL_MODEL_EVERY_MS)
		return;
	vmodel.shader = model.model.shader;
	update_virtual_model(model, disabled_faces_to_copy);
}

bool *VirtualModel::get_disabled_faces_original() {
	await();
	return disabled_faces_for_original;
}

void VirtualModel::clear() {
	cudaMemsetAsync(disabled_faces_for_original, 0, sizeof(bool) * m_allocated_disabled_faces, stream);
	cudaMemsetAsync(disabled_faces_for_virtual, 0, sizeof(bool) * m_allocated_disabled_faces, stream);
	cudaMemsetAsync(vmodel.faces, 0, sizeof(glm::ivec3) * vmodel.n_faces, stream);
	cudaMemsetAsync(vmodel.vertices, 0, sizeof(glm::vec3) * vmodel.n_vertices, stream);
	cudaMemsetAsync(vmodel.normals, 0, sizeof(glm::vec3) * vmodel.n_vertices, stream);
	// cudaMemsetAsync(vmodel.textures, 0, sizeof(glm::vec2) * vmodel.max_texture_index, stream);
}
