//
// Created by dev on 8/27/22.
//

#ifndef COURSE_RENDERER_RASTERIZER_H
#define COURSE_RENDERER_RASTERIZER_H

#include "../zbuffer/zbuffer.h"
#include "synchronizable.h"

class Rasterizer : public Synchronizable {
public:
	Rasterizer() = default;
	void async_rasterize(DrawCallArgs &args, int model_index, Image image, ZBuffer zbuffer);
};

#endif//COURSE_RENDERER_RASTERIZER_H
