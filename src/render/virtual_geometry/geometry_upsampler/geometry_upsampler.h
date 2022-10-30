//
// Created by dev on 10/20/22.
//

#ifndef COURSE_RENDERER_GEOMETRY_UPSAMPLER_H
#define COURSE_RENDERER_GEOMETRY_UPSAMPLER_H

#include "../../draw_caller/synchronizable.h"
#include "../../misc/draw_caller_args.cuh"

class GeometryUpsampler {
	ModelRef &virtual_model;
	cudaStream_t stream;
	int *position = nullptr;
public:
	GeometryUpsampler(ModelRef &virtual_model_, cudaStream_t stream_);
	~GeometryUpsampler();
	void async_upsample_geometry(const ModelDrawCallArgs &model_args, bool *disabled_faces_for_original, bool *disabled_faces_for_virtual);
};

#endif//COURSE_RENDERER_GEOMETRY_UPSAMPLER_H
