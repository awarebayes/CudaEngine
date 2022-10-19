//
// Created by dev on 10/8/22.
//

#ifndef COURSE_RENDERER_MESH_ANALYZER_PUPPETEER_H
#define COURSE_RENDERER_MESH_ANALYZER_PUPPETEER_H

#include "mesh_analyzer.h"
#include <atomic>
#include <memory>
#include <queue>

class MeshAnalyzerPuppeteer : public Synchronizable {
private:
	bool *bad_faces_found_device = nullptr;
	bool *bad_faces_found_host = nullptr;
	int n_analyzers = 0;
	std::vector<std::shared_ptr<MeshAnalyzer>> analyzers;
	std::atomic_bool m_is_analyzing = false;
	std::vector<int> models_in_analysis{};
	std::queue<int> model_analysis_queue{};
	int analyzing_scene_id = -1;

	void copy_bad_faces();
	std::vector<int> get_model_ids_for_analysis(const std::vector<int> &models_with_bad_faces);
public:
	explicit MeshAnalyzerPuppeteer(int n_analyzers);
	~MeshAnalyzerPuppeteer() override;

	int n_calls = 0;

	void analyze_from_queue_BLOCKING(const DrawCallArgs &args, const std::vector<int> &models_with_bad_faces);
	std::vector<int> get_ids_with_bad_faces();
	[[nodiscard]] bool is_analyzing() const { return m_is_analyzing; }
	[[nodiscard]] bool queue_empty() const { return model_analysis_queue.empty(); }
	void enqueue_model(int model_id) { model_analysis_queue.push(model_id); }
};

#endif//COURSE_RENDERER_MESH_ANALYZER_PUPPETEER_H
