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
#include "util/stream_manager.h"
#include <ctime>
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




__device__ void triangle_old(int2 ts[3], Image &image, float4 color) {
	auto t0 = ts[0];
	auto t1 = ts[1];
	auto t2 = ts[2];
	if (t0.y==t1.y && t0.y==t2.y) return; // i dont care about degenerate triangles
	if (t0.y>t1.y) swap(t0, t1);
	if (t0.y>t2.y) swap(t0, t2);
	if (t1.y>t2.y) swap(t1, t2);
	int total_height = t2.y-t0.y;

	auto t0f = float2{float(t0.x), float(t0.y)};
	auto t1f = float2{float(t1.x), float(t1.y)};
	auto t2f = float2{float(t2.x), float(t2.y)};
	auto colori = rgbaFloatToInt(color);

	__syncthreads();

	for (int i=0; i<total_height; i++) {
		bool second_half = i>t1.y-t0.y || t1.y==t0.y;
		int segment_height = second_half ? t2.y-t1.y : t1.y-t0.y;
		float alpha = (float)i/(float)total_height;
		float beta  = (float)(i-(second_half ? t1.y-t0.y : 0))/(float)segment_height; // be careful: with above conditions no division by zero here
		float2 A =               t0f + (t2f-t0f) * alpha;
		float2 B = second_half ? t1f + (t2f-t1f)*beta : t0f + (t1f-t0f)*beta;
		if (A.x>B.x) swap(A, B);
		for (int j=int(A.x); j<=int(B.x); j++) {
			image.set(j, t0.y+i, colori); // attention, due to int casts t0.y+i != A.y
		}
		__syncthreads();
	}
}


template <typename Tp>
__device__ float3 barycentric(float3 *pts, Tp P) {
	auto a = float3{float(pts[2].x-pts[0].x), float(pts[1].x-pts[0].x), float(pts[0].x-P.x)};
	auto b = float3{float(pts[2].y-pts[0].y), float(pts[1].y-pts[0].y), float(pts[0].y-P.y)};
	auto u = cross(a, b);
	float flag = abs(u.z)<1;
	return float3{
	                -1.0f * flag + (1.0f - flag) * (1.f-(u.x+u.y)/u.z),
	                 1.0f * flag + (1.0f - flag) * (u.y/u.z),
	                 1.0f * flag + (1.0f - flag) * (u.x/u.z)
	        };
}

__device__ static float atomicMax(float* address, float val)
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

__device__ void triangle_zbuffer(float3 pts[3], Image &image) {
	float2 bboxmin{float(image.width-1),  float(image.height-1)};
	float2 bboxmax{0., 0.};
	float2 clamp{float(image.width-1), float(image.height-1)};
	for (int i=0; i<3; i++) {
		bboxmin.x = max(0.0f, min(bboxmin.x, pts[i].x));
		bboxmin.y = max(0.0f, min(bboxmin.y, pts[i].y));

		bboxmax.x = min(clamp.x, max(bboxmax.x, pts[i].x));
		bboxmax.y = min(clamp.y, max(bboxmax.y, pts[i].y));
	}

	float3 P{0, 0, 0};

	for (P.x=floor(bboxmin.x); P.x<=bboxmax.x; P.x++) {
		for (P.y=floor(bboxmin.y); P.y<=bboxmax.y; P.y++) {
			auto bc_screen  = barycentric(pts, P);
			float bc_screen_idx[3] = {bc_screen.x, bc_screen.y, bc_screen.z};
			if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0)
				continue;
			P.z = 0;
			for (int i = 0; i < 3; i++)
				P.z += pts[i].z * bc_screen_idx[i];
			atomicMax(&image.zbuffer[int(P.x + P.y * image.width)], P.z);
		}
	}
}


__device__ void triangle(float3 pts[3], Image &image, float4 color) {
	float2 bboxmin{float(image.width-1),  float(image.height-1)};
	float2 bboxmax{0., 0.};
	float2 clamp{float(image.width-1), float(image.height-1)};
	for (int i=0; i<3; i++) {
		bboxmin.x = max(0.0f, min(bboxmin.x, pts[i].x));
		bboxmin.y = max(0.0f, min(bboxmin.y, pts[i].y));

		bboxmax.x = min(clamp.x, max(bboxmax.x, pts[i].x));
		bboxmax.y = min(clamp.y, max(bboxmax.y, pts[i].y));
	}

	auto colori = rgbaFloatToInt(color);
	float3 P{0, 0, 0};

	for (P.x=floor(bboxmin.x); P.x <= bboxmax.x; P.x++) {
		for (P.y=floor(bboxmin.y); P.y <= bboxmax.y; P.y++) {
			auto bc_screen  = barycentric(pts, P);
			float bc_screen_idx[3] = {bc_screen.x, bc_screen.y, bc_screen.z};
			if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0)
				continue;
			P.z = 0;
			for (int i = 0; i < 3; i++)P.z += pts[i].z * bc_screen_idx[i];
			if (image.zbuffer[int(P.x + P.y* image.width)] == P.z) {
				image.set((int)P.x, (int)P.y, colori);
			}
		}
	}
}

__global__ void fill_zbuffer(Image image, ModelRef model) {
	int position = blockIdx.x * blockDim.x + threadIdx.x;
	if (position >= model.n_faces)
		return;
	auto face = model.faces[position];
	float3 screen_coords[3];
	float3 world_coords[3];
	float3 light_dir{0.0, 0.0, -1.0};
	int face_idx[3] = {face.x, face.y, face.z};
	for (int j = 0; j < 3; j++)
	{
		float3 v = model.vertices[face_idx[j]];
		screen_coords[j] = float3{float((v.x + 1.0) * image.width / 2.0), float((v.y + 1.0) * image.height / 2.0), v.z};
		world_coords[j] = v;
	}

	float3 n = cross(world_coords[2] - world_coords[0], world_coords[1] - world_coords[0]);
	n = normalize(n);
	float intensity = dot(n, light_dir);
	if (intensity > 0)
		triangle_zbuffer(screen_coords, image);
}


__global__ void draw_faces(Image image, ModelRef model) {
	int position = blockIdx.x * blockDim.x + threadIdx.x;
	if (position >= model.n_faces)
		return;
	auto face = model.faces[position];
	float3 screen_coords[3];
	float3 world_coords[3];
	float3 light_dir{0.0, 0.0, -1.0};
	int face_idx[3] = {face.x, face.y, face.z};
	for (int j = 0; j < 3; j++)
	{
		float3 v = model.vertices[face_idx[j]];
		screen_coords[j] = float3{float((v.x + 1.0) * image.width / 2.0), float((v.y + 1.0) * image.height / 2.0), v.z};
		world_coords[j] = v;
	}

	float3 n = cross(world_coords[2] - world_coords[0], world_coords[1] - world_coords[0]);
	n = normalize(n);
	float intensity = dot(n, light_dir);
	if (intensity > 0)
		triangle(screen_coords, image,  float4{1.0f, 1.0f, 1.0f, 1.0f} * intensity);
}

double main_cuda_launch(Image &image, StopWatchInterface *timer) {
	auto streams = SingletonCreator<StreamManager>().get();
	auto mp = ModelPoolCreator().get();
	ModelRef ref = mp->get("obj/african_head.myobj");

	// var for kernel computation timing
	// sync host and start kernel computation timer
	double dKernelTime;
	// sync host and start kernel computation timer
	dKernelTime = 0.0;

	clock_t begin = clock();
	sdkResetTimer(&timer);

	streams->prepare_to_render();

	int n_grid = ref.n_faces / 32 + 1;
	int n_block = 32;

	fill_zbuffer<<<n_grid, n_block, 0, streams->render>>>(image, ref);
	draw_faces<<<n_grid, n_block, 0, streams->render>>>(image, ref);

	// sync host and stop computation timer

	checkCudaErrors(cudaStreamSynchronize(streams->render));
	checkCudaErrors(cudaMemsetAsync(image.zbuffer, 0.0f, image.width * image.height * sizeof(float), streams->zreset));
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	printf("%f ms\n", elapsed_secs * 1000);
	dKernelTime = sdkGetTimerValue(&timer);

	return dKernelTime / 1000.;
}
