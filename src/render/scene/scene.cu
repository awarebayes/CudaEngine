//
// Created by dev on 8/28/22.
//
#include "scene.h"

std::shared_ptr<Camera> Scene::get_camera() {
	return camera;
}
std::vector<ModelArgs> &Scene::get_models() {
	return models;
}
DrawCallArgs Scene::get_draw_call_args() {
	DrawCallArgs args = {
			.models = models,
			.base = {
					.light_dir = light_dir,
					.camera_pos = camera->position,
					.look_at = camera->position + camera->get_look_direction(),
					.view_matrix = camera->get_view_matrix(),
					.projection_matrix = camera->get_projection_matrix(),
			},
	};
	return args;
}
float3 &Scene::get_light_dir() {
	return light_dir;
}
