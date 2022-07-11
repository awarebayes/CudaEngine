//
// Created by dev on 7/11/22.
//

#ifndef COURSE_RENDERER_STREAM_MANAGER_H
#define COURSE_RENDERER_STREAM_MANAGER_H

#include <cuda_runtime_api.h>

struct StreamManager
{
	cudaStream_t render{};
	cudaStream_t zreset{};

	StreamManager();
	~StreamManager();
	void prepare_to_render() const;
};

#endif//COURSE_RENDERER_STREAM_MANAGER_H
