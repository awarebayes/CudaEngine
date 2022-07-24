//
// Created by dev on 7/14/22.
//

#include "../inc/matrix.cuh"

__device__ __host__ float3 m2v(const mat<4,4> &m)
{
	float3 result{};
	float w = m.at(3, 0);
	result.x = m.at(0, 0) / w;
	result.y = m.at(1, 0) / w;
	result.z = m.at(2, 0) / w;
	return result;
}

__device__ __host__ mat<4, 1> v2m(const float3 &v)
{
	return {v.x, v.y, v.z, 1.0f};
}


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
