//
// Created by dev on 10/20/22.
//


#include "virtual_geometry_object_manager.h"

VirtualGeometryObjectManager::VirtualGeometryObjectManager(int max_virtual_geometry_objects) {
	virtual_models.reserve(max_virtual_geometry_objects);
	for (int i = 0; i < max_virtual_geometry_objects; i++)
		virtual_models.push_back(std::make_shared<VirtualModel>());
}

void VirtualGeometryObjectManager::accept_single(const ModelDrawCallArgs &model_args, bool *disabled_faces, int vface_count) {
	int id = model_args.scene_object_id;
	if (virtual_models_map.find(id) == virtual_models_map.end())
	{
		for (auto &model: virtual_models)
		{
			if (model->holds_nothing())
			{
				model->accept(model_args, disabled_faces, vface_count);
				virtual_models_map[id] = model;
				break;
			}
		}
	}
	else {
		virtual_models_map[id]->update(model_args, disabled_faces, vface_count);
	}
}

void VirtualGeometryObjectManager::release_unclaimed(const std::vector<int> &model_ids_in_query, const std::vector<int> &culled_ids) {
	if (model_ids_in_query.empty())
		return;
	for (const auto &model: virtual_models)
	{
		if (!model->holds_nothing())
		{
			auto model_id = model->get_model_id();
			if (std::find(model_ids_in_query.begin(), model_ids_in_query.end(), model_id) == model_ids_in_query.end()) {
				model->release();
				virtual_models_map.erase(model_id);
			}
		}
		if (!model->holds_nothing() && std::find(culled_ids.begin(), culled_ids.end(), model->get_model_id()) != culled_ids.end()) {
			virtual_models_map.erase(model->get_model_id());
			model->release();// model is culled
		}
	}
}

std::vector<ModelDrawCallArgs> VirtualGeometryObjectManager::to_args() {
	std::vector<ModelDrawCallArgs> args;
	for (auto &model: virtual_models)
	{
		if (model->holds_nothing())
			continue;
		args.push_back(model->get_virtual_updated());
	}
	return args;
}

bool *VirtualGeometryObjectManager::get_disabled_faces_for_original(int model_id) {
	return virtual_models_map[model_id]->get_disabled_faces_original();
}
