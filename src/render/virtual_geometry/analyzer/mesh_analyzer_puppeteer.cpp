//
// Created by dev on 10/8/22.
//

#include "mesh_analyzer_puppeteer.h"
#include "../../../scene/scene.h"
#include "../../../util/const.h"
#include <cuda_runtime_api.h>

MeshAnalyzerPuppeteer::MeshAnalyzerPuppeteer(int n_analyzers_) : Synchronizable() {
	n_analyzers = n_analyzers_;
	cudaMalloc(reinterpret_cast<void **>(&bad_faces_found_device), sizeof(bool) * n_analyzers);
	cudaMallocHost(reinterpret_cast<void **>(&bad_faces_found_host), sizeof(bool) * n_analyzers);
	for (int i = 0; i < n_analyzers; ++i) {
		analyzers.emplace_back(std::make_shared<MeshAnalyzer>(VIRTUAL_GEOMETRY_VERTICES, 5000));
		analyzers[i]->has_bad_faces = bad_faces_found_device + i;
	}
}

MeshAnalyzerPuppeteer::~MeshAnalyzerPuppeteer() {
	cudaFree(bad_faces_found_device);
	cudaFreeHost(bad_faces_found_host);
}

void MeshAnalyzerPuppeteer::copy_bad_faces() {
	cudaMemcpyAsync(bad_faces_found_host, bad_faces_found_device, sizeof(bool) * n_analyzers, cudaMemcpyDeviceToHost, stream);
}

void MeshAnalyzerPuppeteer::analyze_from_queue_BLOCKING(const DrawCallArgs &args, const std::vector<int> &models_with_bad_faces) {
	assert(!m_is_analyzing);

	if (SceneSingleton().get()->get_time() < 100)
		return;

	m_is_analyzing = true;
	if (args.scene_id != analyzing_scene_id) {
		analyzing_scene_id = args.scene_id;
		model_analysis_queue = std::queue<int>();
		m_is_analyzing = false;
		std::cout << "MeshAnalyzerPuppeteer::analyze_from_queue_BLOCKING: scene changed, clearing queue" << std::endl;
		return;
	}

	cudaMemsetAsync(bad_faces_found_device, 0, sizeof(bool) * n_analyzers, stream);

	models_in_analysis = get_model_ids_for_analysis(models_with_bad_faces);

	for (int i = 0; i < models_in_analysis.size(); ++i) {
		bad_faces_found_host[i] = false;
		analyzers[i]->async_analyze_mesh(args, models_in_analysis[i]);
	}

	for (int i = 0; i < models_in_analysis.size(); ++i) {
		analyzers[i]->await();
	}

	copy_bad_faces();
	await();
	n_calls++;

	analyzing_scene_id = args.scene_id;

	m_is_analyzing = false;
}

std::vector<int> MeshAnalyzerPuppeteer::get_ids_with_bad_faces() {
	assert(!m_is_analyzing);
	std::vector<int> bad_models{};
	for (int i = 0; i < models_in_analysis.size(); ++i) {
		if (bad_faces_found_host[i]) {
			bad_models.push_back(models_in_analysis[i]);
		}
	}
	return bad_models;
}

std::vector<int> MeshAnalyzerPuppeteer::get_model_ids_for_analysis(const std::vector<int> &models_with_bad_faces) {
	assert(!model_analysis_queue.empty());
	assert(models_with_bad_faces.size() < n_analyzers);
	std::vector<int> model_indices = models_with_bad_faces;
	int max_index = std::min(n_analyzers, static_cast<int>(model_analysis_queue.size()));
	max_index - model_indices.size();

	for (int i = 0; i < max_index; ++i) {
		if (std::find(model_indices.begin(), model_indices.end(), model_analysis_queue.front()) == model_indices.end()) {
			model_indices.push_back(model_analysis_queue.front());
			model_analysis_queue.pop();
		}
	}
	return  model_indices;
}
