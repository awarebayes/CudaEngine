//
// Created by dev on 8/28/22.
//

#ifndef COURSE_RENDERER_SCENE_H
#define COURSE_RENDERER_SCENE_H

#include "../../camera/camera.h"
#include "../../kernels/inc/render.cuh"
class Scene {
private:
	std::vector<StoredModel> models{};
	std::shared_ptr<Camera> camera{};
	float3 light_dir{0, 0, 1.5};
public:
	Scene() = default;
	void display_menu();
	void add_model(const StoredModel &model);
	StoredModel &get_model(int index);
	std::shared_ptr<Camera> get_camera();
	void set_camera(const Camera &cam);
	std::vector<ModelArgs> get_models();
	float3 &get_light_dir();
	DrawCallArgs get_draw_call_args();
	int get_n_models();
};

using SceneSingleton = SingletonCreator<Scene>;

#endif//COURSE_RENDERER_SCENE_H
