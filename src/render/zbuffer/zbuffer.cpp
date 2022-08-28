//
// Created by dev on 8/27/22.
//
#include "zbuffer.h"
#include <cuda_runtime_api.h>

OwnedZBuffer::OwnedZBuffer(int height_, int width_) {
	height = height_;
	width = width_;
	cudaMalloc((void**)&zbuffer, width * height * sizeof(float));
}
OwnedZBuffer::~OwnedZBuffer() {
	cudaFree(zbuffer);
}
void ZBuffer::create(Image image) {
	create(image.height, image.width);
}
void ZBuffer::create(int height_, int width_) {
	if (height != height_ || width != width_) {
		height = height_;
		width = width_;
		cudaFree(zbuffer);
		cudaMalloc((void**)&zbuffer, width * height * sizeof(float));
	}
}
