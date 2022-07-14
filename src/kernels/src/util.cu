//
// Created by dev on 7/13/22.
//

#include "../inc/util.cuh"



// Euclidean Distance (x, y, d) = exp((|x - y| / d)^2 / 2)
__device__ float euclideanLen(float4 a, float4 b, float d) {
	float mod = (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y) +
	            (b.z - a.z) * (b.z - a.z);

	return __expf(-mod / (2.f * d * d));
}

__device__ uint rgbaFloatToInt(float4 rgba) {
	rgba.x = __saturatef(fabs(rgba.x));// clamp to [0.0, 1.0]
	rgba.y = __saturatef(fabs(rgba.y));
	rgba.z = __saturatef(fabs(rgba.z));
	rgba.w = __saturatef(fabs(rgba.w));
	return (uint(rgba.w * 255.0f) << 24) | (uint(rgba.z * 255.0f) << 16) |
	       (uint(rgba.y * 255.0f) << 8) | uint(rgba.x * 255.0f);
}

__device__ float4 rgbaIntToFloat(uint c) {
	float4 rgba;
	rgba.x = (c & 0xff) * 0.003921568627f;        //  /255.0f;
	rgba.y = ((c >> 8) & 0xff) * 0.003921568627f; //  /255.0f;
	rgba.z = ((c >> 16) & 0xff) * 0.003921568627f;//  /255.0f;
	rgba.w = ((c >> 24) & 0xff) * 0.003921568627f;//  /255.0f;
	return rgba;
}

__device__ float atomicMax(float* address, float val)
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
