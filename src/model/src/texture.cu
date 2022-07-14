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
void Texture::free() const {
	checkCudaErrors(cudaFree(data));
}
TextureRef Texture::get_ref() const {
	return TextureRef{x, y, n, data};
}
