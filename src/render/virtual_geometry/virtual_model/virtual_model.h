//
// Created by dev on 10/8/22.
//

#ifndef COURSE_RENDERER_VIRTUAL_MODEL_H
#define COURSE_RENDERER_VIRTUAL_MODEL_H

#include "../../misc/draw_caller_args.cuh"
#include <optional>

class VirtualModel {
private:
	std::optional<int> holding_id = std::nullopt;
	std::optional<ModelRef> holding_model = std::nullopt;
	ModelRef vmodel;
	bool *disabled_faces = nullptr;

public:
	VirtualModel();
	~VirtualModel();
	ModelDrawCallArgs to_args();
};

#endif//COURSE_RENDERER_VIRTUAL_MODEL_H
