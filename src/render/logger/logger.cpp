//
// Created by dev on 10/3/22.
//
#include "logger.h"
#include "../../scene/scene.h"
#include "../../util/const.h"
#include "../../util/implot.h"
#include <imgui.h>
#include <imgui_internal.h>
#include <filesystem>

void RenderInterface::draw_widget() {


	if (ImGui::CollapsingHeader("FPS"))
	{
		ImGui::SliderInt("Threads per block", &USE_THREADS, 1, 128);
		ImGui::Text("FPS: %f", fps.back());
		ImGui::PlotLines("Framerate", &fps[0], fps.size(), 0, nullptr, 0.0f, 100.0f, ImVec2(300, 100));

		if (ImGui::Button("Clear average fps"))
		{
			fps.clear();
		}
		if (ImGui::Button("Clear average fps per nthreads"))
		{
			avg_fps_per_nthreads.clear();
		}
		avg_fps_per_nthreads.display("FPS per nthreads");
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
		if (threshold_ptr != nullptr)
		{
			ImGui::SliderInt("Threshold", threshold_ptr, 50, 10000);
		}
	}
}
void RenderInterface::log_fps() {
	fps.push_back(ImGui::GetIO().Framerate);
	if (fps.size() > 100) {
		fps.erase(fps.begin());
	}
	avg_fps_per_nthreads.log(USE_THREADS, ImGui::GetIO().Framerate);
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

RenderInterface::RenderInterface() {
	avg_fps_per_nthreads.log_to("logs/avg_fps_per_nthreads.csv");
}

void AverageLogger::log(float key, float value) {
	if (sum.find(key) == sum.end()) {
		sum[key] = 0;
		count[key] = 0;
	}
	sum[key] += value;
	count[key]++;
	file << std::to_string(key) + "," + std::to_string(value) + "\n";
	if (count[key] % 100 == 0) {
		file << std::flush;
	}
}

float AverageLogger::get_average(float key) {
	return sum[key] / count[key];
}

void AverageLogger::display(const std::string &title)
{
	sorted.clear();
	for (auto &p : sum) {
		sorted.push_back({p.first, p.second / count[p.first]});
	}
	std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) {
		return a.first > b.first;
	});

	keys.clear();
	values.clear();
	for (auto &p : sorted) {
		keys.push_back(p.first);
		values.push_back(p.second);
	}

	if (ImPlot::BeginPlot(title.data())) {
		ImPlot::SetupAxes("x","f(x)");
		ImPlot::PlotLine("sin(x)", (float *)keys.data(), values.data(), sorted.size());
		ImPlot::EndPlot();
	}


}
void AverageLogger::clear() {
	sum.clear();
	count.clear();
	std::filesystem::resize_file(filename, 0);
}
