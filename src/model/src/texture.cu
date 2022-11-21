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

void rgba_to_rgb(unsigned char *rgb, const unsigned char *rgba, int width, int height) {
	for (int i = 0; i < width * height; i++) {
		rgb[3 * i] = rgba[4 * i];
		rgb[3 * i + 1] = rgba[4 * i + 1];
		rgb[3 * i + 2] = rgba[4 * i + 2];
	}
}

Texture::Texture(const std::string &filename) {
	unsigned char *data_host = stbi_load(filename.c_str(), &x, &y, &n, 3);
	assert(data_host != nullptr);
	size_t size = x * y * n * sizeof(unsigned char);
	if (n == 4)
	{
		unsigned char *rgb = new unsigned char[x * y * 3];
		rgba_to_rgb(rgb, data_host, x, y);
		checkCudaErrors(cudaMalloc(&data, x * y * 3 * sizeof(unsigned char)));
		checkCudaErrors(cudaMemcpy(data, rgb, x * y * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
		delete[] rgb;
	}
	else
	{
		assert(n == 3);
		checkCudaErrors(cudaMalloc(&data, size));
		checkCudaErrors(cudaMemcpy(data, data_host, size, cudaMemcpyHostToDevice));
	}
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
