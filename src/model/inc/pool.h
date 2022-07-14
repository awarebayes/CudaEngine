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
public:
	ModelPool() = default;
	~ModelPool() = default;
	ModelRef get(const std::string &path);
	std::shared_ptr<Model> get_mut(const std::string &path);
};

using ModelPoolCreator = SingletonCreator<ModelPool>;

#endif//COURSE_RENDERER_POOL_H
