//
// Created by dev on 8/27/22.
//
#include "../../kernels/inc/render.cuh"
#include "../../kernels/inc/shader_impl.cuh"
#include "../../util/const.h"
#include "rasterizer.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

extern __device__ __constant__ mat<4,4> viewport_matrix;

__device__ void triangle(DrawCallBaseArgs &args, ModelArgs &model_args, int position, Image &image, ZBuffer &zbuffer) {
	auto light_dir = args.light_dir;

	auto &model = model_args.model;
	// auto &model_matrix = model_args.model_matrix;

	glm::mat4 view          = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
	glm::mat4 projection    = glm::mat4(1.0f);
	projection = glm::perspective(glm::radians(45.0f), (float)1920 / (float)1080, 0.1f, 100.0f);
	view       = glm::translate(view, glm::vec3(0.0f, 0.0f, -3.0f));
	glm::mat4 model_matrix = glm::mat4(1.0f);
	model_matrix = glm::translate(model_matrix, glm::vec3(0.0f, 0.0f, 0.0f));
	auto sh = Shader(model, light_dir, projection, view, model_matrix);

	for (int i = 0; i < 3; i++)
		sh.vertex(position, i);

	auto &pts = sh.pts;
	auto &normals = sh.normals;
	auto &textures = sh.textures;

	if (pts[0].y==pts[1].y && pts[0].y==pts[2].y) return;

	float2 bboxmin{float(image.width-1),  float(image.height-1)};
	float2 bboxmax{0., 0.};
	float2 clamp{float(image.width-1), float(image.height-1)};
	for (auto &pt : pts) {
		bboxmin.x = max(0.0f, min(bboxmin.x, pt.x));
		bboxmin.y = max(0.0f, min(bboxmin.y, pt.y));

		bboxmax.x = min(clamp.x, max(bboxmax.x, pt.x));
		bboxmax.y = min(clamp.y, max(bboxmax.y, pt.y));
	}

	float3 P{0, 0, 0};

	int cnt = 0;
	for (P.x=floor(bboxmin.x); P.x <= bboxmax.x; P.x++) {
		for (P.y=floor(bboxmin.y); P.y <= bboxmax.y; P.y++) {
			auto bc_screen  = barycentric(pts, P);
			if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0)
				continue;

			P.z = 0;
			for (int i = 0; i < 3; i++)
				P.z += pts[i].z * at(bc_screen, i);

			if (zbuffer.zbuffer[int(P.x + P.y* image.width)] == P.z) {
				uint color;
				sh.fragment(bc_screen, color);
				image.set((int)P.x, (int)P.y, color);
			}

			cnt++;
			if (cnt > MAX_PIXELS_PER_KERNEL)
				return;
		}
	}

}


__global__ void draw_faces(DrawCallBaseArgs args, ModelArgs model_args, Image image, ZBuffer zbuffer) {

	int position = blockIdx.x * blockDim.x + threadIdx.x;
	auto model = model_args.model;
	if (position >= model.n_faces)
		return;

	auto face = model.faces[position];


	float3 world_coords[3];
	auto look_dir = args.look_at - args.camera_pos;
	for (int j = 0; j < 3; j++)
	{
		float3 v = model.vertices[at(face, j)];
		world_coords[j] = v;
	}

	float3 n = cross(world_coords[2] - world_coords[0], world_coords[1] - world_coords[0]);
	n = normalize(n);
	float intensity = dot(n, look_dir);
	if (intensity > 0)
		triangle(args, model_args, position, image, zbuffer);
}

void Rasterizer::async_rasterize(DrawCallArgs &args, int model_index, Image image, ZBuffer zbuffer)
{

	auto &model_args = args.models[model_index];
	auto &model = model_args.model;
	auto n_grid = model.n_faces / 32 + 1;
	auto n_block = dim3(32);
	draw_faces<<<n_grid, n_block, 0, stream>>>(args.base, model_args, image, zbuffer);
}

