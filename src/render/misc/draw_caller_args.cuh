//
// Created by dev on 10/4/22.
//

#ifndef COURSE_RENDERER_DRAW_CALLER_ARGS_CU
#define COURSE_RENDERER_DRAW_CALLER_ARGS_CU

#include <glm/glm.hpp>
#include "../../camera/camera.h"
#include "../../model/inc/model.h"

struct DrawCallBaseArgs
{
	glm::vec3 light_dir{};
	glm::vec3 camera_pos{};
	glm::vec3 look_at{};
	glm::mat4 view{};
	glm::mat4 projection{};
	glm::vec2 screen_size{1920, 1080};
	Camera *camera_ptr = nullptr;
	float time = 0;
};

struct ModelDrawCallArgs {
	ModelRef model;
	glm::mat4 model_matrix{};
	int id = 0;
	bool *disabled_faces = nullptr;
};

struct DrawCallArgs {
	std::vector<ModelDrawCallArgs> models{};
	DrawCallBaseArgs base{};
	int scene_id = 0;
};

struct SceneObject {
	glm::vec3 position{};
	ModelRef model{};
	ModelDrawCallArgs to_args();
	int id;
};

#endif//COURSE_RENDERER_DRAW_CALLER_ARGS_CU
