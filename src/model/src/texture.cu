//
// Created by dev on 7/13/22.
//
#include "../../util/stb_image.h"
#include "../inc/texture.cuh"
#include <cassert>
#include <helper_cuda.h>

__device__ uchar3 TextureRef::get_uv(float u, float v) const {
	int x_pos = int(u * (float)x);
	int y_pos = int(v * (float)y);
	return data[y_pos * y + x_pos];

}
Texture::Texture(const std::string &filename) {
	unsigned char *data_host = stbi_load(filename.c_str(), &x, &y, &n, 3);
	size_t size = x * y * n;
	assert(n == 3);
	checkCudaErrors(cudaMalloc(&data, size));
	checkCudaErrors(cudaMemcpy(data, data_host, size, cudaMemcpyHostToDevice));
	stbi_image_free(data_host);
}
Texture::~Texture() {
	checkCudaErrors(cudaFree(data));
}
TextureRef Texture::get_ref() const {
	return TextureRef{x, y, n, data};
}
