//
// Created by dev on 10/8/22.
//

#ifndef COURSE_RENDERER_VIRTUAL_GEOMETRY_MANGER_H
#define COURSE_RENDERER_VIRTUAL_GEOMETRY_MANGER_H

#include "../analyzer/mesh_analyzer_puppeteer.h"

class VirtualGeometryManager {
	int n_analyzers = 64;
	int n_virtual_geometry_objects = 16;
	std::shared_ptr<MeshAnalyzerPuppeteer> mesh_analyzer{};
public:
	VirtualGeometryManager();
	void populate_virtual_models(DrawCallArgs& culled_args, const Image &image, const DrawCallArgs& unculled_args);
};

#endif//COURSE_RENDERER_VIRTUAL_GEOMETRY_MANGER_H
