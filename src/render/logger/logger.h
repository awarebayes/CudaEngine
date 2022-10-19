//
// Created by dev on 10/3/22.
//

#ifndef COURSE_RENDERER_LOGGER_H
#define COURSE_RENDERER_LOGGER_H
#include <vector>

class RenderInterface {
protected:
	bool culling_enabled = true;
	std::vector<float> fps;
	std::vector<float> culling_percentage;
	int n_models_before_culling;
	int n_models_after_culling;
public:
	bool is_culling_enabled() const;
	void log_fps();
	void log_before_culling(int n_models);
	void log_after_culling(int n_models);
	void draw_widget();
};

#endif//COURSE_RENDERER_LOGGER_H
