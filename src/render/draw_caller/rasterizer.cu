//
// Created by dev on 8/27/22.
//
#include "../../kernels/inc/render.cuh"
#include "../../kernels/inc/shader_impl.cuh"
#include "../../util/const.h"
#include "rasterizer.h"

extern __device__ __constant__ mat<4,4> viewport_matrix;
extern __device__ mat<4,4> projection_matrix;
extern __device__ mat<4,4> view_matrix;

__device__ void triangle(DrawCallBaseArgs &args, ModelRef &model, int position, Image &image, ZBuffer &zbuffer) {
	auto light_dir = args.light_dir;

	mat<4,4> transform_mat = dot(dot(dot(viewport_matrix, projection_matrix), args.model_matrix), view_matrix);
	auto sh = Shader(model, light_dir);
	sh.uniform_M = transform_mat;

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

	if ((bboxmax.x - bboxmin.x)  * (bboxmax.y - bboxmin.y) > MAX_PIXELS_PER_KERNEL)
		return;

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
		}
	}

}


__global__ void draw_faces(DrawCallBaseArgs args, ModelRef model, Image image, ZBuffer zbuffer) {

	int position = blockIdx.x * blockDim.x + threadIdx.x;
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
		triangle(args, model, position, image, zbuffer);
}

void Rasterizer::async_rasterize(DrawCallArgs &args, ModelRef model, Image image, ZBuffer zbuffer)
{
	auto n_grid = model.n_faces / 32 + 1;
	auto n_block = dim3(32);
	draw_faces<<<n_grid, n_block, 0, stream>>>(args.base, model, image, zbuffer);
}

