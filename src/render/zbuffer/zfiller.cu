//
// Created by dev on 8/27/22.
//
#include "../../shader/all.h"
#include "../../util/const.h"
#include "zbuffer.h"
#include "zfiller.h"
#include <glm/glm.hpp>
#include <helper_math.h>

__device__ void triangle_zbuffer(glm::vec3 pts[3], ZBuffer &zbuffer) {
	glm::vec2 bboxmin{float(zbuffer.width-1),  float(zbuffer.height-1)};
	glm::vec2 bboxmax{0., 0.};
	glm::vec2 clamp{float(zbuffer.width-1), float(zbuffer.height-1)};
	for (int i=0; i<3; i++) {
		bboxmin.x = max(0.0f, min(bboxmin.x, pts[i].x));
		bboxmin.y = max(0.0f, min(bboxmin.y, pts[i].y));

		bboxmax.x = min(clamp.x, max(bboxmax.x, pts[i].x));
		bboxmax.y = min(clamp.y, max(bboxmax.y, pts[i].y));
	}


	glm::vec3 P{0, 0, 0};
	int cnt = 0;
	for (P.x=floor(bboxmin.x); P.x<=bboxmax.x; P.x++) {
		for (P.y=floor(bboxmin.y); P.y<=bboxmax.y; P.y++) {
			P.z = 0;
			auto bc_screen  = barycentric(pts[0], pts[1], pts[2], P);

			auto bc_clip = glm::vec3(bc_screen[0]/pts[0].z, bc_screen[1]/pts[1].z, bc_screen[2]/pts[2].z);
			bc_clip = bc_clip / (bc_clip.x + bc_clip.y + bc_clip.z);

			if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0)
				continue;
			for (int i = 0; i < 3; i++)
				P.z += pts[i].z * bc_clip[i];
			atomicMax(&zbuffer.zbuffer[int(P.x + P.y * zbuffer.width)], P.z);
			cnt++;
			if (cnt > MAX_PIXELS_PER_KERNEL)
				return;
		}
	}
}

template<typename ShaderType>
__global__ void fill_zbuffer(DrawCallBaseArgs args, ModelDrawCallArgs model_args, ZBuffer buffer) {
	int position = blockIdx.x * blockDim.x + threadIdx.x;

	auto &model = model_args.model;
	if (position >= model.n_faces)
		return;

	if (model_args.disabled_faces != nullptr && model_args.disabled_faces[position])
		return;

	auto light_dir = args.light_dir;
	auto sh = BaseShader<ShaderType>(model, light_dir, args.projection, args.view, model_args.model_matrix, args.screen_size, args);

	for (int i = 0; i < 3; i++)
		sh.vertex(position, i, false);


	glm::vec3 look_dir = args.look_at - args.camera_pos;
	glm::vec3 n = cross(sh.pts[2] - sh.pts[0], sh.pts[1] - sh.pts[0]);
	if (glm::dot(look_dir, {0, 0, 1}) > 0)
		n = -n;
	if (dot(n, look_dir) > 0) {
		triangle_zbuffer(sh.pts, buffer);
	}
}

__global__ void set_kernel(ZBuffer buffer, float set_to)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= buffer.width || y >= buffer.height)
		return;
	buffer.zbuffer[x + y * buffer.width] = set_to;
}

void ZFiller::async_zbuf(DrawCallArgs &args, int model_index) {
	auto &model_args = args.models[model_index];
	auto &model = model_args.model;
	auto n_grid = model.n_faces / 32 + 1;
	auto n_block = 32;
	switch (model.shader)
	{
		case RegisteredShaders::Default:
			fill_zbuffer<ShaderDefault><<<n_grid, n_block, 0, stream>>>(args.base, model_args, zbuffer);
			break;
		case RegisteredShaders::Water:
			fill_zbuffer<ShaderWater><<<n_grid, n_block, 0, stream>>>(args.base, model_args, zbuffer);
			break;
		case RegisteredShaders::VGeom:
			fill_zbuffer<ShaderVGeom><<<n_grid, n_block, 0, stream>>>(args.base, model_args, zbuffer);
			break;
	}
}

ZBuffer ZFiller::get_zbuffer() {
	return ZBuffer{.zbuffer=zbuffer.zbuffer, .width=zbuffer.width, .height=zbuffer.height};
}
void ZFiller::async_reset() {
	const dim3 block(16,16);
	const dim3 grid(divUp(zbuffer.width, block.x), divUp(zbuffer.height, block.y));
	set_kernel<<<grid, block, 0, stream>>>(zbuffer, -FLT_MAX);
}

void ZFiller::resize(int height, int width) {
	zbuffer.create(height, width);
}
void ZFiller::resize(Image &image) {
	resize(image.height, image.width);
}
