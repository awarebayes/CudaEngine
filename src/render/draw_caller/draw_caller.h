//
// Created by dev on 8/27/22.
//

#ifndef COURSE_RENDERER_DRAW_CALLER_H
#define COURSE_RENDERER_DRAW_CALLER_H

#include "../../util/singleton.h"
#include "../culler/culler.h"
#include "../image_resetter/image_resetter.h"
#include "../logger/logger.h"
#include "../virtual_geometry/analyzer/mesh_analyzer_puppeteer.h"
#include "../virtual_geometry/manager/virtual_geometry_manger.h"
#include "../zbuffer/zfiller.h"
#include "../zbuffer/zmerger.h"
#include "rasterizer.h"

class DrawCaller
{
private:
	int n_streams = 64;
	std::vector<std::shared_ptr<Rasterizer>> rasterizers{};
	std::vector<std::shared_ptr<ZFiller>> zfillers{};
	std::vector<std::shared_ptr<ZMerger>> z_mergers{};
	std::shared_ptr<ImageResetter> image_resetter{};
	std::shared_ptr<Culler> culler{};
	std::shared_ptr<RenderInterface> interface{};
	std::shared_ptr<VirtualGeometryManager> virtual_geometry_manager{};
public:
	DrawCaller();
	~DrawCaller() = default;
	void draw(DrawCallArgs args, Image &image);
	std::vector<ZBuffer> get_z_buffers();
	ZBuffer get_final_z_buffer();
	bool can_load_scene();
};

using DrawCallerSigleton = SingletonCreator<DrawCaller>;

#endif//COURSE_RENDERER_DRAW_CALLER_H
