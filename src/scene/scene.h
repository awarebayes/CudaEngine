//
// Created by dev on 8/28/22.
//

#ifndef COURSE_RENDERER_SCENE_H
#define COURSE_RENDERER_SCENE_H

#include "../render/misc/draw_caller_args.cuh"
class Scene {
private:
	std::vector<SceneObject> models{};
	std::shared_ptr<Camera> camera{};
	glm::vec3 light_dir{0, 0, 1};
	int id_counter = 0;
	int scene_id = 0;
	int time = 0;
	bool sorted = false;

	std::function <void(Scene &)> on_update = [](Scene &) {};
public:
	Scene() = default;
	void display_menu();
	void add_model(const SceneObject &model);
	void set_light_dir(const glm::vec3 &dir);
	SceneObject &get_model(int index);
	std::shared_ptr<Camera> get_camera();
	void set_camera(const Camera &cam);
	std::vector<ModelDrawCallArgs> get_models();
	glm::vec3 &get_light_dir();
	DrawCallArgs get_draw_call_args();
	int get_n_models();
	void clear();
	void sort_models();
	bool allow_culling = true;
	void tick() { time++; on_update(*this); };
	int get_time() { return time; };

	void set_on_update(std::function <void(Scene &)> on_update) {
		this->on_update = on_update;
	}

};

using SceneSingleton = SingletonCreator<Scene>;

#endif//COURSE_RENDERER_SCENE_H
