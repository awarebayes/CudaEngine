//
// Created by dev on 10/8/22.
//

#include "virtual_geometry_manger.h"
#include <future>

VirtualGeometryManager::VirtualGeometryManager() {
	mesh_analyzer = std::make_shared<MeshAnalyzerPuppeteer>(n_analyzers);
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

		virtual_geometry_object_manager->release_unclaimed(models_with_bad_faces);

		for (auto bad_face_model_id : models_with_bad_faces)
		{
			auto bad_draw_args = *std::find_if(unculled_args.models.begin(), unculled_args.models.end(), [bad_face_model_id](ModelDrawCallArgs const& a) { return a.scene_object_id == bad_face_model_id; });
			virtual_geometry_object_manager->accept(bad_draw_args, mesh_analyzer->get_bad_faces(bad_face_model_id));
		}

		// launch task
		std::async(std::launch::async, &MeshAnalyzerPuppeteer::analyze_from_queue_BLOCKING, this->mesh_analyzer, unculled_args, image, models_with_bad_faces);
	}
}
