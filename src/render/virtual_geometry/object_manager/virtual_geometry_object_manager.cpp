//
// Created by dev on 10/20/22.
//


#include "virtual_geometry_object_manager.h"

VirtualGeometryObjectManager::VirtualGeometryObjectManager(int max_virtual_geometry_objects) {
	virtual_models.reserve(max_virtual_geometry_objects);
	for (int i = 0; i < max_virtual_geometry_objects; i++)
		virtual_models.push_back(std::make_shared<VirtualModel>());
}

void VirtualGeometryObjectManager::accept(const ModelDrawCallArgs &model_args, bool *disabled_faces) {
	int id = model_args.scene_object_id;
	if (virtual_models_map.find(id) == virtual_models_map.end())
	{
		for (auto &model: virtual_models)
		{
			if (model->holds_nothing())
			{
				model->accept(model_args, disabled_faces);
				virtual_models_map[id] = model;
				break;
			}
		}
	}
	else {
		virtual_models_map[id]->accept(model_args, disabled_faces);
	}
}
void VirtualGeometryObjectManager::release(const std::vector<int> &model_ids) {
	for (auto id: model_ids)
	{
		if (virtual_models_map.find(id) != virtual_models_map.end())
		{
			virtual_models_map[id]->release();
			virtual_models_map.erase(id);
		}
		else
		{
			assert(false && "model not found");
		}
	}
}
std::vector<ModelDrawCallArgs> VirtualGeometryObjectManager::to_args() {
std::vector<ModelDrawCallArgs> args;
	for (auto &model: virtual_models)
	{
		if (model->holds_nothing())
			continue;
		args.push_back(model->to_args());
	}
	return args;
}
