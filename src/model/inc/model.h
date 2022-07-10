//
// Created by dev on 7/9/22.
//

#ifndef COURSE_RENDERER_MODEL_H
#define COURSE_RENDERER_MODEL_H

#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "vector_types.h"


struct ModelRef
{
	float3 *vertices{};
	int3 *faces{};
	int n_vertices = 0;
	int n_faces = 0;
};

struct Model {
	float3 *vertices;
	int3 *faces;
	int n_vertices = 0;
	int n_faces = 0;
	explicit Model(const std::string &filename);
	~Model();
	[[nodiscard]] ModelRef get_ref() const;
};

#endif//COURSE_RENDERER_MODEL_H
