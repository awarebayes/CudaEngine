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

#endif//COURSE_RENDERER_KERNELS_UTIL_CUH
