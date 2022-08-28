//
// Created by dev on 8/28/22.
//

#ifndef COURSE_RENDERER_IMAGE_RESETTER_H
#define COURSE_RENDERER_IMAGE_RESETTER_H

#include "../../kernels/inc/render.cuh"
#include "../draw_caller/synchronizable.h"
#include <driver_types.h>
class ImageResetter : public Synchronizable {
private:
	cudaStream_t stream{};
public:
	ImageResetter() = default;
	void async_reset(Image &image);
};

#endif//COURSE_RENDERER_IMAGE_RESETTER_H
