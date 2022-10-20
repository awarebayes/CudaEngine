//
// Created by dev on 10/20/22.
//

#ifndef COURSE_RENDERER_VIRTUAL_GEOMETRY_OBJECT_MANAGER_H
#define COURSE_RENDERER_VIRTUAL_GEOMETRY_OBJECT_MANAGER_H

#include "../virtual_model/virtual_model.h"

class VirtualGeometryObjectManager {
private:
	std::vector<std::shared_ptr<VirtualModel>> virtual_models;
	std::unordered_map<int, std::shared_ptr<VirtualModel>> virtual_models_map;

public:
	explicit VirtualGeometryObjectManager(int max_virtual_geometry_objects);
	void accept(const ModelDrawCallArgs &model_args, bool *disabled_faces);
	void release(const std::vector<int> &model_ids);
	std::vector<ModelDrawCallArgs> to_args();
};

#endif//COURSE_RENDERER_VIRTUAL_GEOMETRY_OBJECT_MANAGER_H
