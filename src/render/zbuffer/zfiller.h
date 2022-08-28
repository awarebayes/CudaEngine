//
// Created by dev on 8/27/22.
//

#ifndef COURSE_RENDERER_ZFILLER_H
#define COURSE_RENDERER_ZFILLER_H

#include "../draw_caller/synchronizable.h"
#include "zbuffer.h"

class ZFiller : public Synchronizable {
private:
	OwnedZBuffer zbuffer{};
public:
	ZFiller() = default;
	void async_zbuf(DrawCallArgs &args, int model_index);
	void async_reset();
	ZBuffer get_zbuffer();
	void resize(int height, int width);
	void resize(Image &image);
};

#endif//COURSE_RENDERER_ZFILLER_H
