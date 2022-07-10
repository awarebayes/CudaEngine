//
// Created by dev on 7/10/22.
//
#include "../inc/pool.h"

ModelRef ModelPool::get(const std::string &path) {
	if (pool.find(path) == pool.end())
		pool.insert(std::make_pair(path, std::make_shared<Model>(path)));
	return pool.at(path)->get_ref();
}

std::shared_ptr<ModelPool> ModelPoolCreator::singleton = std::shared_ptr<ModelPool>(nullptr);

std::shared_ptr<ModelPool> ModelPoolCreator::get() {
	if (!ModelPoolCreator::singleton)
		ModelPoolCreator::singleton = std::make_shared<ModelPool>();
	return ModelPoolCreator::singleton;
}

