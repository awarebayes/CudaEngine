//
// Created by dev on 10/3/22.
//
#include "logger.h"
#include <imgui.h>

void RenderInterface::draw_widget() {
	if (ImGui::CollapsingHeader("FPS"))
	{
		ImGui::Text("FPS: %f", fps.back());
		ImGui::PlotLines("Framerate", &fps[0], fps.size(), 0, nullptr, 0.0f, 100.0f, ImVec2(300, 100));
	}

	if (ImGui::CollapsingHeader("Culling")) {
		ImGui::Checkbox("Enable Culling", &culling_enabled);
		ImGui::Text("Called to draw models: %d", n_models_before_culling);
		ImGui::Text("Drawn after culling: %d", n_models_after_culling);
		ImGui::Text("Culling Percentage: %.1f", culling_percentage.back());
		ImGui::PlotLines("Culling Percentage", culling_percentage.data(), culling_percentage.size(), 0, nullptr, 0.0f, 100.0f, ImVec2(0, 80));
	}
}
void RenderInterface::log_fps() {
	fps.push_back(ImGui::GetIO().Framerate);
	if (fps.size() > 100) {
		fps.erase(fps.begin());
	}
}
void RenderInterface::log_before_culling(int n_models) {
	n_models_before_culling = n_models;
}
void RenderInterface::log_after_culling(int n_models) {
	n_models_after_culling = n_models;
	assert(n_models_before_culling >= n_models_after_culling);
	culling_percentage.push_back((1.0f - (float)n_models_after_culling / (float)n_models_before_culling) * 100.0f);
	if (culling_percentage.size() > 100) {
		culling_percentage.erase(culling_percentage.begin());
	}
}
bool RenderInterface::is_culling_enabled() {
	return culling_enabled;
}
