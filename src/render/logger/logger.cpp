//
// Created by dev on 10/3/22.
//
#include "logger.h"
#include <imgui.h>

void RenderLogger::draw_widget() {
	ImGui::Begin("Render Logger");
	ImGui::Text("FPS: %f", fps.back());
	ImGui::PlotHistogram("Framerate", &fps[0], fps.size(), 0, nullptr, 0.0f, 100.0f, ImVec2(300, 100));
	ImGui::Text("Called to draw %d models, drawn after culling %d", n_models_before_culling, n_models_after_culling);
	ImGui::Text("Culling percentage: %f", culling_percentage.back());
	ImGui::PlotHistogram("Culling percentage", &culling_percentage[0], culling_percentage.size(), 0, nullptr, 0.0f, 100.0f, ImVec2(300, 100));
	ImGui::End();
}
void RenderLogger::log_fps() {
	fps.push_back(ImGui::GetIO().Framerate);
	if (fps.size() > 100) {
		fps.erase(fps.begin());
	}
}
void RenderLogger::log_before_culling(int n_models) {
	n_models_before_culling = n_models;
}
void RenderLogger::log_after_culling(int n_models) {
	n_models_after_culling = n_models;
	assert(n_models_before_culling >= n_models_after_culling);
	culling_percentage.push_back((1.0f - (float)n_models_after_culling / (float)n_models_before_culling) * 100.0f);
	if (culling_percentage.size() > 100) {
		culling_percentage.erase(culling_percentage.begin());
	}
}
