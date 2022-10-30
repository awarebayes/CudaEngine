//
// Created by dev on 10/3/22.
//
#include "logger.h"
#include "../../scene/scene.h"
#include <imgui.h>
#include <imgui_internal.h>

void RenderInterface::draw_widget() {
	if (ImGui::CollapsingHeader("FPS"))
	{
		ImGui::Text("FPS: %f", fps.back());
		ImGui::PlotLines("Framerate", &fps[0], fps.size(), 0, nullptr, 0.0f, 100.0f, ImVec2(300, 100));
	}

	if (ImGui::CollapsingHeader("Culling")) {
		if (!SceneSingleton().get()->allow_culling)
		{
			ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
			ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
			bool f = false;
			ImGui::Checkbox("Enable Culling", &f);
			ImGui::PopItemFlag();
			ImGui::PopStyleVar();
		}
		else
		{
			ImGui::Checkbox("Enable Culling", &culling_enabled);
		}
		ImGui::Text("Called to draw models: %d", n_models_before_culling);
		ImGui::Text("Drawn after culling: %d", n_models_after_culling);
		ImGui::Text("Culling Percentage: %.1f", culling_percentage.back());
		ImGui::PlotLines("Culling Percentage", culling_percentage.data(), culling_percentage.size(), 0, nullptr, 0.0f, 100.0f, ImVec2(0, 80));
	}
	if (ImGui::CollapsingHeader("Virtual Geometry")) {
		ImGui::Checkbox("Enable Virtual Geometry", &virtual_geometry_enabled);
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
bool RenderInterface::is_culling_enabled() const {
	if (SceneSingleton().get()->allow_culling)
		return culling_enabled;
	return false;
}
bool RenderInterface::is_virtual_geometry_enabled() const {
	return virtual_geometry_enabled;
}
void RenderInterface::register_vgeometry_manager(VirtualGeometryManager &manager) {
	threshold_ptr = &manager.get_threshold_mut();
}
