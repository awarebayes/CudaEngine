//
// Created by dev on 7/14/22.
//

#include "../inc/matrix.cuh"
#include <helper_math.h>


__device__ __host__ mat<4, 4> viewport(int x, int y, int w, int h, int depth)
{
	mat<4, 4> result = identity_matrix<4>();
	result.at(0, 0) = (float)w / 2.0f;
	result.at(1, 1) = (float)h / 2.0f;
	result.at(2, 2) = (float)depth / 2.0f;
	result.at(0, 3) = (float)x + (float)w / 2.0f;
	result.at(1, 3) = (float)y + (float)h / 2.0f;
	result.at(2, 3) = (float)depth / 2.0f;
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

__device__ __host__ mat<4,4> lookat(float3 eye, float3 center, float3 up)
{
	float3 z = normalize(eye - center);
	float3 x = normalize(cross(up,z));
	float3 y = normalize(cross(z,x));
	mat<4,4> Minv = identity_matrix<4>();
	mat<4,4> Tr   = identity_matrix<4>();
	for (int i=0; i<3; i++) {
		Minv.at(0, i) = at(x, i);
		Minv.at(1, i) = at(y, i);
		Minv.at(2, i) = at(z, i);
		Tr.at(i, 3) = -at(eye, i);
	}
	return dot(Minv, Tr);
}

__host__ mat<4,4> offset(float3 xyz)
{
	mat<4,4> result = identity_matrix<4>();
	result.at(0, 3) = xyz.x;
	result.at(1, 3) = xyz.y;
	result.at(2, 3) = xyz.z;
	return result;
}
