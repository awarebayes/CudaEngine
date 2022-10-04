//
// Created by dev on 7/10/22.
//
#include "../inc/pool.h"

ModelRef ModelPool::get(const std::string &path) {
	if (pool.find(path) == pool.end())
		throw std::runtime_error("Model not found in pool");
	return pool.at(path)->get_ref();
}

std::shared_ptr<Model> ModelPool::get_mut(const std::string &path) {
	if (pool.find(path) == pool.end())
		throw std::runtime_error("Model not found in pool");
	return pool.at(path);
}

void ModelPool::load_all_from_obj_file(const std::string &filename) {
	tinyobj::ObjReaderConfig reader_config;
	reader_config.mtl_search_path = "./";
	tinyobj::ObjReader reader;

	if (!reader.ParseFromFile(filename, reader_config)) {
		if (!reader.Error().empty()) {
			std::cerr << "TinyObjReader: " << reader.Error();
		}
		exit(1);
	}

	for (int i = 0; i < reader.GetShapes().size(); i++) {
		auto model = std::make_shared<Model>(reader, i);
		pool.insert(std::make_pair(filename + ":" + reader.GetShapes()[i].name, model));
	}
}
void ModelPool::assign_single_texture_to_obj_file(const std::string &obj_filename, const std::string &texture_filename) {
	tinyobj::ObjReaderConfig reader_config;
	reader_config.mtl_search_path = "./";
	tinyobj::ObjReader reader;

	if (!reader.ParseFromFile(obj_filename, reader_config)) {
		if (!reader.Error().empty()) {
			std::cerr << "TinyObjReader: " << reader.Error();
		}
		exit(1);
	}

	for (int i = 0; i < reader.GetShapes().size(); i++) {
		auto model = get_mut(obj_filename + ":" + reader.GetShapes()[i].name);
		model->load_texture(texture_filename);
	}
}
std::vector<ModelRef> ModelPool::get_all() {
	std::vector<ModelRef> models;
	for (auto &model : pool) {
		models.push_back(model.second->get_ref());
	}
	return models;
}
void ModelPool::clear() {
	pool.clear();
}
