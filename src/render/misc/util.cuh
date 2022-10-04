//
// Created by dev on 7/13/22.
//

#ifndef COURSE_RENDERER_KERNELS_UTIL_CUH
#define COURSE_RENDERER_KERNELS_UTIL_CUH

#include "../../../Common/helper_functions.h"
#include "../../camera/camera.h"
#include "../../model/inc/model.h"
#include <glm/glm.hpp>


template <typename Tp>
__device__ __forceinline__ glm::vec3 barycentric(glm::vec3 a, glm::vec3 b, glm::vec3 c, Tp P)
{
	auto m = glm::vec3{float(c.x-a.x), float(b.x-a.x), float(a.x-P.x)};
	auto n = glm::vec3{float(c.y-a.y), float(b.y-a.y), float(a.y-P.y)};
	auto u = glm::cross(n, m);
	return glm::vec3{1.f-(u.x+u.y)/u.z, u.y/u.z, u.x/u.z};
}

template<typename T>
__device__ __forceinline__ void swap(T &x, T &y)
{
	T temp = x;
	x = y;
	y = temp;
}

__device__ float euclideanLen(float4 a, float4 b, float d);
__device__ float4 rgbaIntToFloat(uint c);

__device__ __forceinline__ uint rgbaFloatToInt(float4 rgba) {
	rgba.x = __saturatef(fabs(rgba.x));// clamp to [0.0, 1.0]
	rgba.y = __saturatef(fabs(rgba.y));
	rgba.z = __saturatef(fabs(rgba.z));
	rgba.w = __saturatef(fabs(rgba.w));
	return (uint(rgba.w * 255.0f) << 24) | (uint(rgba.z * 255.0f) << 16) |
	       (uint(rgba.y * 255.0f) << 8) | uint(rgba.x * 255.0f);
}


__device__ __forceinline__ float atomicMax(float* address, float val)
{
	int* address_as_i = (int*) address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
		                  __float_as_int(::fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

#endif//COURSE_RENDERER_KERNELS_UTIL_CUH
