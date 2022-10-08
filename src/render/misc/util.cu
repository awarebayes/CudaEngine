//
// Created by dev on 7/13/22.
//

#include "../../render/misc/draw_caller_args.cuh"
#include "image.cuh"
#include "util.cuh"
#include <glm/ext/matrix_transform.hpp>


// Euclidean Distance (x, y, d) = exp((|x - y| / d)^2 / 2)
__device__ float euclideanLen(float4 a, float4 b, float d) {
	float mod = (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y) +
	            (b.z - a.z) * (b.z - a.z);

	return __expf(-mod / (2.f * d * d));
}

__device__ float4 rgbaIntToFloat(uint c) {
	float4 rgba;
	rgba.x = (c & 0xff) * 0.003921568627f;        //  /255.0f;
	rgba.y = ((c >> 8) & 0xff) * 0.003921568627f; //  /255.0f;
	rgba.z = ((c >> 16) & 0xff) * 0.003921568627f;//  /255.0f;
	rgba.w = ((c >> 24) & 0xff) * 0.003921568627f;//  /255.0f;
	return rgba;
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

ModelDrawCallArgs SceneObject::to_args() {
	return ModelDrawCallArgs{
	        model,
	        glm::translate(glm::mat4(1.0f), position),
	        id,
	};
}
