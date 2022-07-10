//
// Created by dev on 7/10/22.
//

#ifndef COURSE_RENDERER_POOL_H
#define COURSE_RENDERER_POOL_H

#include "model.h"
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
};

class ModelPoolCreator
{
private:
	static std::shared_ptr<ModelPool> singleton;
public:
	std::shared_ptr<ModelPool> get();
};

#endif//COURSE_RENDERER_POOL_H
