#include "../../model/inc/model.h"
#include "../../render/zbuffer/zbuffer.h"
#include "../../util/stream_manager.h"
#include "../inc/shader_impl.cuh"
#include <glm/ext/matrix_transform.hpp>
#include <helper_math.h>


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

ModelArgs StoredModel::to_args() {
	return ModelArgs{
	        glm::translate(glm::mat4(1.0f), position),
			model,
	};
}
