//
// Created by dev on 7/13/22.
//
#include "../../util/stb_image.h"
#include "../inc/texture.cuh"
#include <cassert>
#include <helper_cuda.h>

__device__ uchar3 TextureRef::get_uv(float u, float v) const {
	v = 1 - v;
	int x_pos = int(u * (float)x);
	int y_pos = int(v * (float)y);

	x_pos = x_pos % x;
	y_pos = y_pos % y;

	return data[x_pos + y_pos * x];
}
Texture::Texture(const std::string &filename) {
	unsigned char *data_host = stbi_load(filename.c_str(), &x, &y, &n, 3);
	size_t size = x * y * n * sizeof(unsigned char);
	assert(n == 3);
	checkCudaErrors(cudaMalloc(&data, size));
	checkCudaErrors(cudaMemcpy(data, data_host, size, cudaMemcpyHostToDevice));
	stbi_image_free(data_host);
}

TextureRef Texture::get_ref() const {
	return TextureRef{x, y, n, data};
}
Texture Texture::get_default() {
	Texture texture;
	texture.x = 1;
	texture.y = 1;
	texture.n = 3;
	checkCudaErrors(cudaMalloc(&texture.data, sizeof(uchar3)));
	uchar3 color{255, 255, 255};
	checkCudaErrors(cudaMemcpy(texture.data, &color, sizeof(uchar3), cudaMemcpyHostToDevice));
	return texture;
}

Texture::~Texture() {
	cudaFree(data);
}
