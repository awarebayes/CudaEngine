//
// Created by dev on 10/8/22.
//

#include "virtual_geometry_manger.h"
#include <future>

VirtualGeometryManager::VirtualGeometryManager() {
	mesh_analyzer = std::make_shared<MeshAnalyzerPuppeteer>(n_analyzers, threshold);
	virtual_geometry_object_manager = std::make_shared<VirtualGeometryObjectManager>(n_virtual_geometry_objects);
}

void VirtualGeometryManager::populate_virtual_models(DrawCallArgs &culled_args, const Image &image, const DrawCallArgs &unculled_args)
{
	if (!mesh_analyzer->is_analyzing())
	{
		// virtual geometry analysis
		if (mesh_analyzer->queue_empty())
		{
			for(auto& model : culled_args.models)
				mesh_analyzer->enqueue_model(model.model.id);
		}
		// get analysis results
		auto models_with_bad_faces = mesh_analyzer->get_ids_with_bad_faces();

		std::vector<int> culled_ids = {};
		for (const auto &any_model: unculled_args.models)
		{
			bool on_scene = false;
			for (const auto& model_in_scene : culled_args.models)
			{
				if (model_in_scene.model.id == any_model.model.id)
				{
					on_scene = true;
					break;
				}
			}
			if (!on_scene)
				culled_ids.push_back(any_model.model.id);
		}

		virtual_geometry_object_manager->release_unclaimed(models_with_bad_faces, culled_ids);

		for (auto bad_face_model_id : models_with_bad_faces)
		{
			auto bad_draw_args = *std::find_if(unculled_args.models.begin(), unculled_args.models.end(), [bad_face_model_id](ModelDrawCallArgs const& a) { return a.scene_object_id == bad_face_model_id; });
			virtual_geometry_object_manager->accept_single(bad_draw_args, mesh_analyzer->get_disabled_faces(bad_face_model_id), mesh_analyzer->get_vface_count(bad_face_model_id));
		}
		// launch task
		std::async(std::launch::async, &MeshAnalyzerPuppeteer::analyze_from_queue_BLOCKING, this->mesh_analyzer, unculled_args, image, models_with_bad_faces);
	}

	for (auto& i: virtual_geometry_object_manager->to_args())
	{
		auto original_id_model = std::find_if(culled_args.models.begin(), culled_args.models.end(), [i](ModelDrawCallArgs const& a) { return a.scene_object_id == i.scene_object_id; });
		if (original_id_model == culled_args.models.end())
			continue;
		original_id_model->disabled_faces = virtual_geometry_object_manager->get_disabled_faces_for_original(i.scene_object_id);
		i.model_matrix = original_id_model->model_matrix;
		culled_args.models.push_back(i);
	}
}
