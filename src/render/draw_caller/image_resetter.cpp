//
// Created by dev on 8/28/22.
//
#include "image_resetter.h"

void ImageResetter::async_reset(Image &image) {
	cudaMemsetAsync((void *)image.pixels, 0, image.width * image.height * sizeof(uint), stream);
}
