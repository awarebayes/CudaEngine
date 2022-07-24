#include "../../model/inc/model.h"
#include "../../model/inc/pool.h"
#include "../../util/stream_manager.h"
#include "../inc/render.cuh"
#include "../inc/matrix.cuh"
#include "../inc/util.cuh"
#include <ctime>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_math.h>
#include <thrust/fill.h>

__device__ __constant__ mat<4,4> viewport_matrix{};
__device__ __constant__ mat<4,4> projection_matrix{};


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
			P.z = 0;
			auto bc_screen  = barycentric(pts, P);
			float bc_screen_idx[3] = {bc_screen.x, bc_screen.y, bc_screen.z};
			if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0)
				continue;
			for (int i = 0; i < 3; i++)
				P.z += pts[i].z * bc_screen_idx[i];
			atomicMax(&image.zbuffer[int(P.x + P.y * image.width)], P.z);
		}
	}
}


__device__ void triangle(DrawCallArgs &args, const int index[3], Image &image) {
	auto &model = args.model;
	auto &light_dir = args.light_dir;
	float3 pts[3];
	float3 normals[3];
	float2 textures[3];

	for (int i = 0; i < 3; i++)
	{
		float3 v = model.vertices[index[i]];
		normals[i] = model.normals[index[i]];
		textures[i] = model.textures[index[i]];
		pts[i] = m2v(dot(dot(viewport_matrix, projection_matrix), v2m(v)));
	}

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

	for (P.x=floor(bboxmin.x); P.x <= bboxmax.x; P.x++) {
		for (P.y=floor(bboxmin.y); P.y <= bboxmax.y; P.y++) {
			auto bc_screen  = barycentric(pts, P);
			float bc_screen_idx[3] = {bc_screen.x, bc_screen.y, bc_screen.z};
			if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0)
				continue;

			P.z = 0;
			for (int i = 0; i < 3; i++)
				P.z += pts[i].z * bc_screen_idx[i];

			if (image.zbuffer[int(P.x + P.y* image.width)] == P.z) {
				float3 N{};
				float2 T{};
				for (int i = 0; i < 3; i++)
				{
					N += normals[i] * bc_screen_idx[i];
					T += textures[i] * bc_screen_idx[i];
				}
				uchar3 color_u = model.texture.get_uv(T.x, T.y);
				float4 color = float4{float(color_u.x), float(color_u.y), float(color_u.z), 255.0f} / 255.0f;

				float4 colorf = color * dot(light_dir, N);
				colorf.w = 1.0f;
				auto colori = rgbaFloatToInt(colorf);
				image.set((int)P.x, (int)P.y, colori);
			}
		}
	}
}

__global__ void fill_zbuffer(DrawCallArgs args) {


	auto &model = args.model;
	auto &image = args.image;
	int position = blockIdx.x * blockDim.x + threadIdx.x;

	auto temp = m2v(dot(dot(viewport_matrix, projection_matrix), v2m({0.5, 0.5, 0.5})));
	printf("RES: %f %f %f \n", temp.x, temp.y, temp.z);

	if (position >= model.n_faces)
		return;
	auto face = model.faces[position];
	float3 screen_coords[3];
	float3 world_coords[3];
	float3 look_dir{0.0, 0.0, -1.0};
	int face_idx[3] = {face.x, face.y, face.z};
	for (int j = 0; j < 3; j++)
	{
		float3 v = model.vertices[face_idx[j]];
		screen_coords[j] = m2v(dot(dot(viewport_matrix, projection_matrix), v2m(v)));
		world_coords[j] = v;
	}

	float3 n = cross(world_coords[2] - world_coords[0], world_coords[1] - world_coords[0]);
	n = normalize(n);
	float intensity = dot(n, look_dir);
	if (intensity > 0)
		triangle_zbuffer(screen_coords, image);
}


__global__ void draw_faces(DrawCallArgs args) {
	auto &model = args.model;
	auto &image = args.image;

	int position = blockIdx.x * blockDim.x + threadIdx.x;
	if (position >= model.n_faces)
		return;
	auto face = model.faces[position];
	float3 world_coords[3];
	auto &look_dir = args.look_dir;
	int vertex_idx[3] = {face.x, face.y, face.z};
	for (int j = 0; j < 3; j++)
	{
		float3 v = model.vertices[vertex_idx[j]];
		world_coords[j] = v;
	}

	float3 n = cross(world_coords[2] - world_coords[0], world_coords[1] - world_coords[0]);
	n = normalize(n);
	float intensity = dot(n, look_dir);
	if (intensity > 0)
		triangle(args, vertex_idx, image);
}


void render_init(int width, int height)
{
	int depth = 255;
	mat<4,4> ViewPort = viewport(width/8, height/8, width*3/4, height*3/4, depth);
	cudaMemcpyToSymbol(
	        viewport_matrix,
	        &ViewPort,
	        sizeof(mat<4,4>)
	        );
}

void update_device_parameters(const DrawCallArgs &args)
{
	mat<4,4> Projection = identity_matrix<4>();

	Projection.at(3, 2) = -1.f / args.camera_pos.z;
	cudaMemcpyToSymbol(
	        projection_matrix,
	        &Projection,
	        sizeof(mat<4,4>)
	        );
}

double main_cuda_launch(const DrawCallArgs &args, StopWatchInterface *timer) {
	auto streams = SingletonCreator<StreamManager>().get();

	// var for kernel computation timing
	// sync host and start kernel computation timer
	double dKernelTime;
	// sync host and start kernel computation timer
	dKernelTime = 0.0;

	clock_t begin = clock();
	sdkResetTimer(&timer);

	update_device_parameters(args);
	streams->prepare_to_render();

	auto &model = args.model;
	auto &image = args.image;
	int n_grid = model.n_faces / 32 + 1;
	int n_block = 32;

	fill_zbuffer<<<n_grid, n_block, 0, streams->render>>>(args);
	draw_faces<<<n_grid, n_block, 0, streams->render>>>(args);

	checkCudaErrors(cudaStreamSynchronize(streams->render));
	thrust::fill(thrust::device, image.zbuffer, image.zbuffer + image.width * image.height, -FLT_MAX);
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	printf("%f ms\n", elapsed_secs * 1000);
	dKernelTime = sdkGetTimerValue(&timer);

	return dKernelTime / 1000.;
}