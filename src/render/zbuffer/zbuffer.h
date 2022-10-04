//
// Created by dev on 8/27/22.
//

#ifndef COURSE_RENDERER_ZBUFFER_H
#define COURSE_RENDERER_ZBUFFER_H

#include "../misc/draw_caller_args.cuh"
#include "../misc/image.cuh"

struct ZBuffer
{
	float *zbuffer{};
	int width{};
	int height{};

	void create(Image image);
	void create(int height_, int width_);
};

struct OwnedZBuffer : ZBuffer
{
	OwnedZBuffer(int height, int width);
	OwnedZBuffer() = default;
	~OwnedZBuffer();
};

#endif//COURSE_RENDERER_ZBUFFER_H
