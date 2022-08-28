//
// Created by dev on 8/28/22.
//
#include "zmerger.h"
#include <cassert>
#include <helper_math.h>

__global__ void max_merge_kernel(ZBuffer z1, ZBuffer z2)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= z1.width || y >= z2.height)
		return;

	int pos = x + y * z1.width;
	z1.zbuffer[pos] = max(z1.zbuffer[pos], z2.zbuffer[pos]);
}

void ZMerger::async_merge(ZBuffer &z1, ZBuffer &z2) {
	assert(z1.width == z2.width);
	assert(z1.height == z2.height);
	assert(z1.zbuffer != nullptr);
	assert(z2.zbuffer != nullptr);
	assert(z1.zbuffer != z2.zbuffer);
	const dim3 block(16,16);
	const dim3 grid(divUp(z1.width, block.x), divUp(z1.height, block.y));
	max_merge_kernel<<<grid, block, 0, stream>>>(z1, z2);
}

void parallel_reduce_merge_helper(std::vector<std::shared_ptr<ZMerger>> &mergers, std::vector<ZBuffer> &zbuffers, int active, int step, int stride) {
	if (active <= 1) return;

	for (int i = 0; i < active; i += step) {
		if (i + stride < active) {
			mergers[i]->async_merge(zbuffers[i], zbuffers[i + stride]);
		}
	}

	for (int i = 0; i < active; i += step) {
		if (i + stride < active) {
			mergers[i]->await();
		}
	}
	if (step < active)
		parallel_reduce_merge_helper(mergers, zbuffers, active, step * 2, stride * 2);
}

void parallel_reduce_merge(std::vector<std::shared_ptr<ZMerger>> &mergers, std::vector<ZBuffer> &zbuffers, int active)
{
	assert(mergers.size() == zbuffers.size());
	assert(mergers.size() > 0);
	assert(zbuffers.size() > 0);
	if (active == 1)
		return;
	parallel_reduce_merge_helper(mergers, zbuffers, active, 2, 1);
}
