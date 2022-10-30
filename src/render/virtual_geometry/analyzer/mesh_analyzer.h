//
// Created by dev on 10/6/22.
//

#ifndef COURSE_RENDERER_MESH_ANALYZER_H
#define COURSE_RENDERER_MESH_ANALYZER_H

#include "../../draw_caller/synchronizable.h"
#include "../../misc/draw_caller_args.cuh"
#include "../../misc/image.cuh"
#include "../../zbuffer/zbuffer.h"

class MeshAnalyzer : public Synchronizable
{
	friend class MeshAnalyzerPuppeteer;
private:
	int capacity = 0;
	int size = 0;

	bool *face_mask = nullptr;
	int *new_vfaces_count = nullptr;
	float *area = nullptr;
	int &area_threshold;

public:
	explicit MeshAnalyzer(int capacity, int &threshold);
	~MeshAnalyzer() override;
	void async_analyze_mesh(const DrawCallArgs &args, const Image &image, int model_index);
};

#endif//COURSE_RENDERER_MESH_ANALYZER_H
