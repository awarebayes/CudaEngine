//
// Created by dev on 10/8/22.
//

#include "mesh_analyzer_puppeteer.h"
#include "../../../scene/scene.h"
#include "../../../util/const.h"
#include <cuda_runtime_api.h>

MeshAnalyzerPuppeteer::MeshAnalyzerPuppeteer(int n_analyzers_, int &area_threshold_) : area_threshold(area_threshold_), Synchronizable() {
	n_analyzers = n_analyzers_;
	cudaMalloc(reinterpret_cast<void **>(&new_vfaces_count_device), sizeof(int) * n_analyzers);
	cudaMallocHost(reinterpret_cast<void **>(&new_vfaces_found_host), sizeof(int) * n_analyzers);
	for (int i = 0; i < n_analyzers; ++i) {
		analyzers.emplace_back(std::make_shared<MeshAnalyzer>(VIRTUAL_GEOMETRY_VERTICES, area_threshold));
		analyzers[i]->new_vfaces_count = new_vfaces_count_device + i;
	}
}

MeshAnalyzerPuppeteer::~MeshAnalyzerPuppeteer() {
	cudaFree(new_vfaces_count_device);
	cudaFreeHost(new_vfaces_found_host);
}

void MeshAnalyzerPuppeteer::copy_new_vfaces_found() {
	cudaMemcpyAsync(new_vfaces_found_host, new_vfaces_count_device, sizeof(int) * n_analyzers, cudaMemcpyDeviceToHost, stream);
}

void MeshAnalyzerPuppeteer::analyze_from_queue_BLOCKING(const DrawCallArgs &args, const Image &image, const std::vector<int> &models_with_bad_faces) {
	assert(!m_busy);

	if (SceneSingleton().get()->get_time() < 3)
		return;

	m_busy = true;
	if (args.scene_id != analyzing_scene_id) {
		analyzing_scene_id = args.scene_id;
		model_analysis_queue = std::queue<int>();
		m_busy = false;

		std::cout << "MeshAnalyzerPuppeteer::analyze_from_queue_BLOCKING: scene changed, clearing queue" << std::endl;
		return;
	}

	memset(new_vfaces_found_host, 0, sizeof(int) * n_analyzers);
	cudaMemsetAsync(new_vfaces_count_device, 0, sizeof(int) * n_analyzers, stream);

	models_in_analysis = get_model_ids_for_analysis(models_with_bad_faces);

	for (int i = 0; i < models_in_analysis.size(); ++i) {
		new_vfaces_found_host[i] = 0;
		analyzers[i]->async_analyze_mesh(args, image, models_in_analysis[i]);
	}

	for (int i = 0; i < models_in_analysis.size(); ++i) {
		analyzers[i]->await();
	}

	copy_new_vfaces_found();
	await();

	n_calls++;
	analyzing_scene_id = args.scene_id;

	m_busy = false;
}

std::vector<int> MeshAnalyzerPuppeteer::get_ids_with_bad_faces() {
	assert(!m_busy);
	std::vector<int> bad_models{};
	for (int i = 0; i < models_in_analysis.size(); ++i) {
		if (new_vfaces_found_host[i]) {
			bad_models.push_back(models_in_analysis[i]);
		}
	}
	return bad_models;
}

std::vector<int> MeshAnalyzerPuppeteer::get_model_ids_for_analysis(const std::vector<int> &models_with_bad_faces) {
	if (model_analysis_queue.empty())
		return models_with_bad_faces;
	assert(models_with_bad_faces.size() < n_analyzers);
	std::vector<int> model_indices = models_with_bad_faces;
	int max_index = std::min(n_analyzers, static_cast<int>(model_analysis_queue.size()));

	for (int i = 0; i < max_index; ++i) {
		if (std::find(model_indices.begin(), model_indices.end(), model_analysis_queue.front()) == model_indices.end()) {
			model_indices.push_back(model_analysis_queue.front());
			model_analysis_queue.pop();
		}
	}
	return  model_indices;
}

bool *MeshAnalyzerPuppeteer::get_disabled_faces(int id) {
	for (int i = 0; i < models_in_analysis.size(); ++i) {
		if (models_in_analysis[i] == id) {
			return analyzers[i]->face_mask;
		}
	}
}

int MeshAnalyzerPuppeteer::get_vface_count(int id)
{
	await();
	int index = -1;
	for (int i = 0; i < models_in_analysis.size(); ++i) {
		if (models_in_analysis[i] == id) {
			index = i;
			break;
		}
	}
	assert(index != -1);
	return new_vfaces_found_host[index];
}
void MeshAnalyzerPuppeteer::reset() {
	analyzing_scene_id = -1;
	model_analysis_queue = std::queue<int>();
}
