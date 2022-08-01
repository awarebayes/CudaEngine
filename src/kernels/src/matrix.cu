//
// Created by dev on 7/14/22.
//

#include "../inc/matrix.cuh"
#include <helper_math.h>

#define GLM_FORCE_CUDA 1
#include <cuda.h>
#include <glm/glm.hpp>

__device__ __host__ glm::mat4 viewport(int x, int y, int w, int h, int depth)
{
	glm::mat4 result = glm::mat4(1.0f);
	result[0][0] = (float)w / 2.0f;
	result[1][1] = (float)h / 2.0f;
	result[2][2] = (float)depth / 2.0f;
	result[0][3] = (float)x + (float)w / 2.0f;
	result[1][3] = (float)y + (float)h / 2.0f;
	result[2][3] = (float)depth / 2.0f;
	return result;
}


__device__ __host__ void dbg_print(const mat<4, 1> &mat)
{
                printf(
	                    "DBG PRINT\n"
                        "%f\n"
                        "%f\n"
                        "%f\n"
                        "%f\n\n",
	                    mat.at(0, 0),
	                    mat.at(0, 1),
	                    mat.at(0, 2),
	                    mat.at(0, 3)
                );
}

__device__ __host__ void dbg_print(const mat<4, 4> &mat)
{
	printf(
		"DBG PRINT\n"
		"%f %f %f %f\n"
		"%f %f %f %f\n"
		"%f %f %f %f\n"
		"%f %f %f %f\n\n",
		mat.at(0, 0), mat.at(0, 1), mat.at(0, 2), mat.at(0, 3),
		mat.at(1, 0), mat.at(1, 1), mat.at(1, 2), mat.at(1, 3),
		mat.at(2, 0), mat.at(2, 1), mat.at(2, 2), mat.at(2, 3),
		mat.at(3, 0), mat.at(3, 1), mat.at(3, 2), mat.at(3, 3)
	);
}

__device__ __host__ glm::mat4 lookat(glm::vec3 eye, glm::vec3 center, glm::vec3 up)
{
	glm::vec3 z = normalize(eye - center);
	glm::vec3 x = normalize(cross(up,z));
	glm::vec3 y = normalize(cross(z,x));
	glm::mat4 Minv = glm::mat4(1.0f);
	glm::mat4 Tr   = glm::mat4(1.0f);
	for (int i=0; i<3; i++) {
		Minv[0][i] = at(x, i);
		Minv[1][i] = at(y, i);
		Minv[2][i] = at(z, i);
		Tr[i][3] = -at(eye, i);
	}
	return Minv * Tr;
}