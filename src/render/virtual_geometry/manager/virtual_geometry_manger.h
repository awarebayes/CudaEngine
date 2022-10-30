//
// Created by dev on 10/8/22.
//

#ifndef COURSE_RENDERER_VIRTUAL_GEOMETRY_MANGER_H
#define COURSE_RENDERER_VIRTUAL_GEOMETRY_MANGER_H

#include "../analyzer/mesh_analyzer_puppeteer.h"
#include "../object_manager/virtual_geometry_object_manager.h"
#include <condition_variable>

class VirtualGeometryManager {
private:
	friend class RenderInterface;
	int n_analyzers = 64;
	int n_virtual_geometry_objects = 16;
	int threshold = 5000;
	std::shared_ptr<MeshAnalyzerPuppeteer> mesh_analyzer{};
	std::shared_ptr<VirtualGeometryObjectManager> virtual_geometry_object_manager{};
public:
	VirtualGeometryManager();
	void populate_virtual_models(DrawCallArgs& culled_args, const Image &image, const DrawCallArgs& unculled_args);
	int &get_threshold_mut() { return threshold; }
};

#endif//COURSE_RENDERER_VIRTUAL_GEOMETRY_MANGER_H
