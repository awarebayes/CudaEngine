//
// Created by dev on 10/8/22.
//

#include "virtual_geometry_manger.h"
#include <future>

VirtualGeometryManager::VirtualGeometryManager() {
	mesh_analyzer = std::make_shared<MeshAnalyzerPuppeteer>(n_analyzers);
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
		// launch task
		std::async(std::launch::async, &MeshAnalyzerPuppeteer::analyze_from_queue_BLOCKING, this->mesh_analyzer, unculled_args, image, models_with_bad_faces);
	}
}
