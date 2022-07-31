//
// Created by dev on 7/13/22.
//

#ifndef COURSE_RENDERER_KERNELS_UTIL_CUH
#define COURSE_RENDERER_KERNELS_UTIL_CUH

template<typename T>
__device__ __forceinline__ void swap(T &x, T &y)
{
	T temp = x;
	x = y;
	y = temp;
}

__device__ float euclideanLen(float4 a, float4 b, float d);
__device__ uint rgbaFloatToInt(float4 rgba);
__device__ float4 rgbaIntToFloat(uint c);
__device__ float atomicMax(float* address, float val);

__device__ __forceinline__ uint rgbaFloatToInt(float4 rgba) {
	rgba.x = __saturatef(fabs(rgba.x));// clamp to [0.0, 1.0]
	rgba.y = __saturatef(fabs(rgba.y));
	rgba.z = __saturatef(fabs(rgba.z));
	rgba.w = __saturatef(fabs(rgba.w));
	return (uint(rgba.w * 255.0f) << 24) | (uint(rgba.z * 255.0f) << 16) |
	       (uint(rgba.y * 255.0f) << 8) | uint(rgba.x * 255.0f);
}

#endif//COURSE_RENDERER_KERNELS_UTIL_CUH
