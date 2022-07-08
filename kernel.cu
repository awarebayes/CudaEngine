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

#include <helper_cuda.h>// CUDA device initialization helper functions
#include <helper_functions.h>
#include <helper_math.h>


// Euclidean Distance (x, y, d) = exp((|x - y| / d)^2 / 2)
__device__ float euclideanLen(float4 a, float4 b, float d) {
  float mod = (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y) +
              (b.z - a.z) * (b.z - a.z);

  return __expf(-mod / (2.f * d * d));
}

__device__ uint rgbaFloatToInt(float4 rgba) {
  rgba.x = __saturatef(fabs(rgba.x));  // clamp to [0.0, 1.0]
  rgba.y = __saturatef(fabs(rgba.y));
  rgba.z = __saturatef(fabs(rgba.z));
  rgba.w = __saturatef(fabs(rgba.w));
  return (uint(rgba.w * 255.0f) << 24) | (uint(rgba.z * 255.0f) << 16) |
         (uint(rgba.y * 255.0f) << 8) | uint(rgba.x * 255.0f);
}

__device__ float4 rgbaIntToFloat(uint c) {
  float4 rgba;
  rgba.x = (c & 0xff) * 0.003921568627f;          //  /255.0f;
  rgba.y = ((c >> 8) & 0xff) * 0.003921568627f;   //  /255.0f;
  rgba.z = ((c >> 16) & 0xff) * 0.003921568627f;  //  /255.0f;
  rgba.w = ((c >> 24) & 0xff) * 0.003921568627f;  //  /255.0f;
  return rgba;
}

// column pass using coalesced global memory reads
__global__ void d_parametric_circle(uint *od, int w, int h, int x_0, int y_0, int radius, int total_pixels) {
	int position = blockIdx.x * blockDim.x + threadIdx.x;
	if (position >= total_pixels)
		return;

	float percent = (float)position / (float)total_pixels;
	float rad = M_PI * 2.0f * percent;
	int x = int((float)x_0 + cos(rad) * (float)radius);
	int y = int((float)y_0 + sin(rad) * (float)radius);

	if (x >= w or x < 0)
		return;
	if (y >= h or y < 0)
		return;

	auto &target_pixel = od[y * w + x];
	// get float4 inetnsity: rgbaIntToFloat(target_pixel);
	target_pixel = rgbaFloatToInt(float4{1.0f, 1.0f, 1.0f, 1.0f});
}

void bresenhamCircle(uint *od, int w, int h, int x_0, int y_0, int radius)
{
	int circumference = ceil((float)radius * 2.0f * M_PI);
	d_parametric_circle<<<circumference / 32, 32>>>(od, w, h, x_0, y_0, radius, circumference);
}


// RGBA version
double main_cuda_launch(uint *dDest, int width, int height, StopWatchInterface *timer) {
    // var for kernel computation timing
    double dKernelTime;

	// sync host and start kernel computation timer
	dKernelTime = 0.0;
	checkCudaErrors(cudaDeviceSynchronize());
	sdkResetTimer(&timer);


	bresenhamCircle(dDest, width, height, width/2, height/2, width/4);

	// sync host and stop computation timer
	checkCudaErrors(cudaDeviceSynchronize());
	dKernelTime += sdkGetTimerValue(&timer);

  return (dKernelTime / 1000.);
}
