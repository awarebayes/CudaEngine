//
// Created by dev on 7/9/22.
//
#include "../inc/model.h"
#include <fstream>
#include <helper_cuda.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

Model::Model(const std::string &filename) : vertices(), faces() {
	std::ifstream in;
	in.open(filename, std::ifstream::in);
	if (in.fail()) throw std::runtime_error("Bad file: " + std::string(filename));
	std::string line;

	std::vector<float3> vertices_host{};
	std::vector<int3> faces_host{};

	while (!in.eof()) {
		std::getline(in, line);
		std::stringstream ss(line);
		ss.str(line);
		std::string kind;
		ss >> kind;

		if (kind == "i")
		{
			ss >> n_vertices >> n_faces;
			vertices_host.reserve(n_vertices);
			faces_host.reserve(n_faces);
		} else if (kind == "v") {
			float3 v;
			ss >> v.x >> v.y >> v.z;
			vertices_host.push_back(v);
		} else if (kind == "f") {
			int3 f;
			ss >> f.x >> f.y >> f.z;
			f.x--;
			f.y--;
			f.z--;
			faces_host.push_back(f);
		}
	}
	assert(n_vertices == vertices_host.size());
	assert(n_faces == faces_host.size());
	checkCudaErrors(cudaMalloc((void**)(&vertices), sizeof(float3) * n_vertices));
	checkCudaErrors(cudaMalloc((void**)(&faces), sizeof(int3) * n_faces));
	checkCudaErrors(cudaMemcpy(vertices, vertices_host.data(), n_vertices * sizeof(float3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(faces, faces_host.data(), n_faces * sizeof(int3), cudaMemcpyHostToDevice));
	std::cerr << "# Model loaded with v# " << vertices_host.size() << " f# "  << faces_host.size() << std::endl;
}

Model::~Model() {
	checkCudaErrors(cudaFree((void*)faces));
	checkCudaErrors(cudaFree((void*)vertices));
}
ModelRef Model::get_ref() const {
	return ModelRef{ vertices, faces, n_vertices, n_faces };
}
