//
// Created by dev on 8/28/22.
//
#include "../shader/registered_shaders.h"
#include "scene.h"
#include <GL/freeglut.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <imgui.h>

std::shared_ptr<Camera> Scene::get_camera() {
	return camera;
}
std::vector<ModelDrawCallArgs> Scene::get_models() {
	auto model_args = std::vector<ModelDrawCallArgs>(models.size());
	for (int i = 0; i < models.size(); i++) {
		model_args[i] = models[i].to_args();
		if (override_shader_enabled)
			model_args[i].model.shader = override_shader;
	}
	return model_args;
}

DrawCallArgs Scene::get_draw_call_args() {

	assert(sorted);
	auto look_dir = camera->get_look_direction();

	DrawCallArgs args = {
			.models = get_models(),
			.base = {
					.light_dir = glm::vec3(light_dir.x, light_dir.y, light_dir.z),
					.camera_pos = camera->position,
					.look_at = camera->position + camera->get_look_direction(),
					.view = camera->get_view_matrix(),
					.projection = camera->get_projection_matrix(),
	                .screen_size = camera->get_screen_size(),
	                .camera_ptr = camera.get(),
	                .time = (float)glutGet(GLUT_ELAPSED_TIME),
			},
	        .scene_id = scene_id,
	};
	return args;
}

glm::vec3 &Scene::get_light_dir() {
	return light_dir;
}
void Scene::set_camera(const Camera &cam) {
	camera = std::make_shared<Camera>(cam);
}

std::string thousandSeparator(size_t n)
{
	std::string ans = {};

	// Convert the given integer
	// to equivalent string
	std::string num = std::to_string(n);

	// Initialise count
	int count = 0;

	// Traverse the string in reverse
	int i = num.size() - 1;
	for (;i >= 0; i--) {
		count++;
		ans.push_back(num[i]);

		// If three characters
		// are traversed
		if (count == 3) {
			ans.push_back(',');
			count = 0;
		}
	}

	// Reverse the string to get
	// the desired output
	reverse(ans.begin(), ans.end());

	// If the given string is
	// less than 1000
	if (ans.size() % 4 == 0) {
		// Remove ','
		ans.erase(ans.begin());
	}

	return ans;
}


void Scene::display_menu() {
	if (ImGui::CollapsingHeader("Scene")) {
		ImGui::SliderFloat3("Light dir", &light_dir.x, -1, 1);

		camera->display_menu();

		size_t vertices = 0;
		size_t triangles = 0;
		for (auto &model : models) {
			vertices += model.model.n_vertices;
			triangles += model.model.n_faces;
		}

		ImGui::Text("Non virtual statistics");
		ImGui::Text("Vertices: %s", thousandSeparator(vertices).c_str());
		ImGui::Text("Triangles: %s", thousandSeparator(triangles).c_str());

		ImGui::Checkbox("Override shader", &override_shader_enabled);
		if (override_shader_enabled) {
			const char* current_item = registered_shaders_string.at(override_shader).data();
			if (ImGui::BeginCombo("Shader", current_item))
			{
				for (const auto &shader : registered_shaders_enum) {
					bool is_selected = (current_item == shader.first);
					if (ImGui::Selectable(shader.first.data(), is_selected)) {
						current_item = shader.first.data();
						override_shader = shader.second;
					}
					if (is_selected) {
						ImGui::SetItemDefaultFocus();
					}
				}

				ImGui::EndCombo();
			}
		}

		if (ImGui::CollapsingHeader("Models")) {
			for (int i = 0; i < models.size(); i++) {
				if (ImGui::TreeNode((void *) (intptr_t) i, "Model %d", i)) {
					ImGui::SliderFloat3("Position", &models[i].position.x, -10, 10);
					const char* current_item = registered_shaders_string.at(models[i].model.shader).data();
					vertices += models[i].model.n_vertices;
					triangles += models[i].model.n_faces;
					if (ImGui::BeginCombo("Shader", current_item))
					{
						for (const auto &shader : registered_shaders_enum) {
							bool is_selected = (current_item == shader.first);
							if (ImGui::Selectable(shader.first.data(), is_selected)) {
								current_item = shader.first.data();
								models[i].model.shader = shader.second;
							}
							if (is_selected) {
								ImGui::SetItemDefaultFocus();
							}
						}

						ImGui::EndCombo();
					}
					ImGui::TreePop();
				}
			}
		}

	}
}
void Scene::add_object(const SceneObject &model) {
	SceneObject model_copy = model;
	model_copy.id = id_counter++;
	models.push_back(model_copy);
	sorted = false;
	scene_id += 1;
}
SceneObject &Scene::get_model(int index) {
	return models[index];
}

int  Scene::get_n_models() {
	return models.size();
}
void Scene::set_light_dir(const glm::vec3 &dir) {
	light_dir = dir;
}
void Scene::clear() {
	this->models.clear();
	id_counter = 0;
	scene_id += 1;
	allow_culling = true;
	time = 0;
	on_update = [](Scene &){};
}
void Scene::sort_models() {
	std::sort(models.begin(), models.end(), [](const SceneObject &a, const SceneObject &b) {
		return a.id < b.id;
	});
	sorted = true;
}
