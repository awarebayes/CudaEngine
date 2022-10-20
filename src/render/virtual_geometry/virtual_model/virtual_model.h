//
// Created by dev on 10/8/22.
//

#ifndef COURSE_RENDERER_VIRTUAL_MODEL_H
#define COURSE_RENDERER_VIRTUAL_MODEL_H

#include "../../misc/draw_caller_args.cuh"
#include <optional>

class VirtualModel {
private:
	std::optional<int> scene_object_id = std::nullopt;
	ModelRef vmodel;
	bool *disabled_faces = nullptr;
	void free();

public:
	VirtualModel();
	~VirtualModel();

	void accept(ModelDrawCallArgs model, bool *disabled_faces);
	void release();
	int get_model_id();
	bool holds_nothing() { return !scene_object_id.has_value(); };
	ModelDrawCallArgs to_args();
};

#endif//COURSE_RENDERER_VIRTUAL_MODEL_H
