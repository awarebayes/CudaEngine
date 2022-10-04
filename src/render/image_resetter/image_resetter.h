//
// Created by dev on 8/28/22.
//

#ifndef COURSE_RENDERER_IMAGE_RESETTER_H
#define COURSE_RENDERER_IMAGE_RESETTER_H

#include "../draw_caller/synchronizable.h"
#include "../misc/draw_caller_args.cuh"
#include "../misc/image.cuh"
#include <driver_types.h>
class ImageResetter : public Synchronizable {
public:
	ImageResetter() : Synchronizable() {}
	void async_reset(Image &image);
};

#endif//COURSE_RENDERER_IMAGE_RESETTER_H
