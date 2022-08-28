//
// Created by dev on 8/28/22.
//
#include "synchronizable.h"
#include <cuda_runtime_api.h>
#include <helper_cuda.h>

Synchronizable::Synchronizable() {
	cudaStreamCreate(&stream);
}
Synchronizable::~Synchronizable() {
	cudaStreamDestroy(stream);
}
void Synchronizable::await() {
	checkCudaErrors(cudaStreamSynchronize(stream));
}

