//
// Created by dev on 7/10/22.
//

#ifndef COURSE_RENDERER_POOL_H
#define COURSE_RENDERER_POOL_H

#include "model.h"
#include "../../util/singleton.h"
#include <string>
#include <unordered_map>

class ModelPool
{
private:
	std::unordered_map<std::string, std::shared_ptr<Model>> pool{};
	std::shared_ptr<Texture> default_texture{};
	int id_counter = 0;
public:
	ModelPool();
	~ModelPool() = default;
	void clear();

	ModelRef get(const std::string &name);
	std::shared_ptr<Model> get_mut(const std::string &name);

	void load_all_from_obj_file(const std::string &filename, const std::string &mtl_search_path = "./", const std::string &texture_search_path = "./");
	void assign_single_texture_to_obj_file(const std::string &obj_filename, const std::string &texture_filename);

	std::vector<std::string> get_all_names();

	std::vector<ModelRef> get_all();
};

using ModelPoolCreator = SingletonCreator<ModelPool>;

#endif//COURSE_RENDERER_POOL_H
