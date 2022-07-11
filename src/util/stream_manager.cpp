//
// Created by dev on 7/11/22.
//
#include "stream_manager.h"
#include <helper_cuda.h>

StreamManager::StreamManager()
{
	cudaStreamCreate(&render);
	cudaStreamCreate(&zreset);
}

StreamManager::~StreamManager()
{
	cudaStreamDestroy(render);
	cudaStreamDestroy(zreset);
}
void StreamManager::prepare_to_render() const {
	checkCudaErrors(cudaStreamSynchronize(zreset));
}
