//
// Created by dev on 10/4/22.
//

#include "../model/inc/pool.h"
#include "scene.h"
#include "scene_loader.h"
void load_rungholt()
{
	auto mp = ModelPoolCreator().get();
	mp->load_all_from_obj_file("obj/rungholt.obj");
	mp->assign_single_texture_to_obj_file("obj/rungholt.obj", "obj/rungholt.tga");

	auto scene = SceneSingleton().get();
	scene->clear();

	Camera camera;
	camera.position = glm::vec3{0, 100, 0};
	camera.yaw = -90;
	camera.pitch = 0;

	for (const auto ref: mp->get_all()) {
		scene->add_model(StoredModel{glm::vec3{0, -100, -50}, ref});
	}

	scene->set_camera(camera);
}

void load_test_single_head()
{

	auto mp = ModelPoolCreator().get();
	mp->load_all_from_obj_file("obj/african_head.obj");
	mp->assign_single_texture_to_obj_file("obj/african_head.obj", "obj/african_head_diffuse.tga");

	auto scene = SceneSingleton().get();

	Camera camera;
	camera.position = glm::vec3{-3, -1.5, 5};
	camera.yaw = -90;
	camera.pitch = 0;

	scene->set_light_dir(glm::vec3{0, 0, 1});

	const auto ref =  mp->get("obj/african_head.obj:head");
	scene->add_model(StoredModel{glm::vec3{0, 0, -5}, ref});


	scene->set_camera(camera);
}

void load_test_many()
{
	auto mp = ModelPoolCreator().get();
	mp->load_all_from_obj_file("obj/african_head.obj");
	mp->assign_single_texture_to_obj_file("obj/african_head.obj", "obj/african_head_diffuse.tga");

	auto scene = SceneSingleton().get();
	scene->clear();

	Camera camera;
	camera.position = glm::vec3{0, -1, 12};
	camera.yaw = -90;
	camera.pitch = 0;

	scene->set_light_dir(glm::vec3{0, 0, 1});

	auto ref = mp->get("obj/african_head.obj:head");
	scene->add_model(StoredModel{glm::vec3{0, 0, -5}, ref});
	for (int i = 0; i < 100; i++)
		for (int j = 0; j < 100; j++)
			scene->add_model(StoredModel{glm::vec3{float(i-50) * 1.5f, float(j / 2), -10.0f * j - 10}, ref});


	scene->set_camera(camera);
}



void load_scene_diablo()
{

	auto mp = ModelPoolCreator().get();
	mp->load_all_from_obj_file("obj/diablo_pose/diablo_pose.obj");
	mp->assign_single_texture_to_obj_file("obj/diablo_pose/diablo_pose.obj", "obj/diablo_pose/diablo_pose_diffuse.tga");
	auto scene = SceneSingleton().get();
	scene->clear();

	Camera camera;
	camera.position = glm::vec3{-1.5, -1, 2};
	camera.yaw = -90;
	camera.pitch = 0;

	scene->set_light_dir(glm::vec3{0, 0, 1});

	auto ref = mp->get("obj/diablo_pose/diablo_pose.obj:objDiablo3");
	scene->add_model(StoredModel{glm::vec3{0, 0, -5}, ref});

	scene->set_camera(camera);
}


void load_scene_water()
{

	auto mp = ModelPoolCreator().get();
	mp->load_all_from_obj_file("obj/water/water.obj");
	mp->assign_single_texture_to_obj_file("obj/water/water.obj", "obj/water/water.jpg");
	auto scene = SceneSingleton().get();
	scene->clear();

	Camera camera;
	camera.position = glm::vec3{-0.36, -0.34, 4};
	camera.yaw = -90;
	camera.pitch = -25;

	scene->set_light_dir(glm::vec3{0, 1, 0});

	auto mut = mp->get_mut("obj/water/water.obj:Plane");
	mut->shader_type = 'w';

	auto ref = mp->get("obj/water/water.obj:Plane");

	scene->add_model(StoredModel{glm::vec3{0, -2, 0}, ref});

	scene->set_camera(camera);
}

void register_predefined_scenes()
{
	auto scene_loader = SceneLoaderSingleton().get();
	scene_loader->register_load_scene("test_single_head", load_test_single_head);
	scene_loader->register_load_scene("test_many_heads", load_test_many);
	scene_loader->register_load_scene("diablo", load_scene_diablo);
	scene_loader->register_load_scene("water", load_scene_water);
	// scene_loader->register_load_scene("rungholt", load_rungholt);

	scene_loader->register_load_scene("default", load_scene_water);
}
