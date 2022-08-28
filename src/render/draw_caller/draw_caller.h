//
// Created by dev on 8/27/22.
//

#ifndef COURSE_RENDERER_DRAW_CALLER_H
#define COURSE_RENDERER_DRAW_CALLER_H

#include "../../util/singleton.h"
#include "../image_resetter/image_resetter.h"
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
public:
	DrawCaller();
	~DrawCaller() = default;
	void draw(DrawCallArgs &args, Image &image);
	std::vector<ZBuffer> get_z_buffers();
	ZBuffer get_final_z_buffer();
};

using DrawCallerSigleton = SingletonCreator<DrawCaller>;

#endif//COURSE_RENDERER_DRAW_CALLER_H
