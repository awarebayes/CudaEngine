//
// Created by dev on 10/8/22.
//

#ifndef COURSE_RENDERER_MESH_ANALYZER_PUPPETEER_H
#define COURSE_RENDERER_MESH_ANALYZER_PUPPETEER_H

#include "mesh_analyzer.h"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <queue>

class MeshAnalyzerPuppeteer : public Synchronizable {
private:
	int *bad_faces_found_device = nullptr;
	int *bad_faces_found_host = nullptr;
	int n_analyzers = 0;
	std::vector<std::shared_ptr<MeshAnalyzer>> analyzers;
	std::vector<int> models_in_analysis{};
	std::queue<int> model_analysis_queue{};
	std::atomic_bool m_busy = false;
	int analyzing_scene_id = -1;
	const size_t max_queue_size = 1024;

	void copy_bad_faces();
	std::vector<int> get_model_ids_for_analysis(const std::vector<int> &models_with_bad_faces);
public:
	explicit MeshAnalyzerPuppeteer(int n_analyzers);
	~MeshAnalyzerPuppeteer() override;

	int n_calls = 0;

	void analyze_from_queue_BLOCKING(const DrawCallArgs &args, const Image &image, const std::vector<int> &models_with_bad_faces);
	std::vector<int> get_ids_with_bad_faces();
	[[nodiscard]] bool queue_empty() const { return model_analysis_queue.empty(); }
	bool *get_bad_faces(int id);
	bool is_analyzing() { return m_busy; }
	void enqueue_model(int model_id) {
		if (model_analysis_queue.size() < max_queue_size)
			model_analysis_queue.push(model_id);
	}
};

#endif//COURSE_RENDERER_MESH_ANALYZER_PUPPETEER_H