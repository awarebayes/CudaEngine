//
// Created by dev on 10/20/22.
//

#ifndef COURSE_RENDERER_VIRTUAL_GEOMETRY_OBJECT_MANAGER_H
#define COURSE_RENDERER_VIRTUAL_GEOMETRY_OBJECT_MANAGER_H

#include "../virtual_model/virtual_model.h"
#include <atomic>

class VirtualGeometryObjectManager {
private:
	std::vector<std::shared_ptr<VirtualModel>> virtual_models;
	std::unordered_map<int, std::shared_ptr<VirtualModel>> virtual_models_map;
	std::atomic_bool m_busy = false;
public:
	explicit VirtualGeometryObjectManager(int max_virtual_geometry_objects);
	void accept_single(const ModelDrawCallArgs &model_args, bool *disabled_faces);
	void release_unclaimed(const std::vector<int> &model_ids_in_query, const std::vector<int> &culled_ids);
	std::vector<ModelDrawCallArgs> to_args();
	bool *get_disabled_faces_for_original(int model_id);
};

#endif//COURSE_RENDERER_VIRTUAL_GEOMETRY_OBJECT_MANAGER_H
