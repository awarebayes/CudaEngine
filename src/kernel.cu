/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "kernel.cuh"
#include "model/inc/model.h"
#include "model/inc/pool.h"
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_math.h>


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

__global__ void d_parametric_circle(Image image, int x_0, int y_0, int radius, int total_pixels) {
	int position = blockIdx.x * blockDim.x + threadIdx.x;
	if (position >= total_pixels)
		return;

	float percent = (float) position / (float) total_pixels;
	float rad = M_PI * 2.0f * percent;
	int x = int((float) x_0 + cos(rad) * (float) radius);
	int y = int((float) y_0 + sin(rad) * (float) radius);

	if (x >= image.width or x < 0)
		return;
	if (y >= image.height or y < 0)
		return;

	auto &target_pixel = image.pixels[y * image.width + x];
	target_pixel = rgbaFloatToInt(float4{1.0f, 1.0f, 1.0f, 1.0f});
}

void parametricCircle(Image &image, int x_0, int y_0, int radius) {
	int circumference = ceil((float) radius * 2.0f * M_PI);
	int n_grid = circumference / 32 + 1;
	int n_block = 32;
	d_parametric_circle<<<n_grid, n_block>>>(image, x_0, y_0, radius, circumference);
}

template<typename T>
__device__ void swap(T &x, T &y)
{
	T temp = x;
	x = y;
	y = temp;
}

__device__ void line(Image &image, int x0, int y0, int x1, int y1) {
	bool steep = false;
	if (std::abs(x0-x1)<std::abs(y0-y1)) {
		swap(x0, y0);
		swap(x1, y1);
		steep = true;
	}
	if (x0>x1) {
		swap(x0, x1);
		swap(y0, y1);
	}
	__syncthreads();

	uint color = rgbaFloatToInt(float4{1.0f, 1.0f, 1.0f, 1.0f});

	for (int x=x0; x<=x1; x++) {
		float t = (x-x0)/(float)(x1-x0);
		int y = y0*(1.-t) + y1*t;
		int x_draw = y * steep + x * (1 - steep);
		int y_draw = x * steep + y * (1 - steep);
		image.set(x_draw, y_draw, color);
	}
}

__device__ void triangle(Image &image, int2 t0, int2 t1, int2 t2, float4 color)
{
	if (t0.y==t1.y && t0.y==t2.y) return; // i dont care about degenerate triangles
	if (t0.y>t1.y) swap(t0, t1);
	if (t0.y>t2.y) swap(t0, t2);
	if (t1.y>t2.y) swap(t1, t2);
	int total_height = t2.y-t0.y;
	uint colori = rgbaFloatToInt(color);
	for (int i=0; i<total_height; i++) {
		bool second_half = i>t1.y-t0.y || t1.y==t0.y;
		int segment_height = second_half ? t2.y-t1.y : t1.y-t0.y;
		float alpha = (float)i/total_height;
		float beta  = (float)(i-(second_half ? t1.y-t0.y : 0))/segment_height; // be careful: with above conditions no division by zero here
		int2 A =               t0 + (t2-t0)*alpha;
		int2 B = second_half ? t1 + (t2-t1)*beta : t0 + (t1-t0)*beta;
		if (A.x>B.x) swap(A, B);
		for (int j=A.x; j<=B.x; j++) {
			image.set(j, t0.y+i, colori); // attention, due to int casts t0.y+i != A.y
		}
	}
}

__global__ void draw_faces(Image image, ModelRef model) {
	int position = blockIdx.x * blockDim.x + threadIdx.x;
	if (position >= model.n_faces)
		return;
	auto face = model.faces[position];
	int2 screen_coords[3];
	float3 world_coords[3];
	int face_idx[3] = {face.x, face.y, face.z};
	for (int j = 0; j < 3; j++)
	{
		float3 v0 = model.vertices[face_idx[j]];
		float3 v1 = model.vertices[face_idx[(j + 1) % 3]];
		int x0 = (v0.x + 1.0f) * image.width / 2.0f;
		int y0 = (v0.y + 1.0f) * image.height / 2.0f;
		int x1 = (v1.x + 1.0f) * image.width / 2.0f;
		int y1 = (v1.y + 1.0f) * image.height / 2.0f;
		line(image, x0, y0, x1, y1);
		// printf("Line (%d %d) (%d %d)\n", x0, y0, x1, y1);
		__syncthreads();
	}
}


double main_cuda_launch(Image &image, StopWatchInterface *timer) {
	auto mp = ModelPoolCreator().get();
	ModelRef ref = mp->get("obj/african_head.myobj");


	// var for kernel computation timing
	double dKernelTime;
	// sync host and start kernel computation timer
	dKernelTime = 0.0;
	checkCudaErrors(cudaDeviceSynchronize());
	sdkResetTimer(&timer);

	int n_grid = ref.n_faces / 32 + 1;
	int n_block = 32;
	draw_faces<<<n_grid, n_block>>>(image, ref);

	// sync host and stop computation timer
	checkCudaErrors(cudaDeviceSynchronize());
	dKernelTime = sdkGetTimerValue(&timer);

	return dKernelTime / 1000.;
}
