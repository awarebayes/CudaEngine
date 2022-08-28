//
// Created by dev on 8/28/22.
//

#ifndef COURSE_RENDERER_ZMERGER_H
#define COURSE_RENDERER_ZMERGER_H

#include "../draw_caller/synchronizable.h"
#include "zbuffer.h"
#include <driver_types.h>
#include <memory>
#include <vector>

class ZMerger : public Synchronizable {
private:
	cudaStream_t stream{};
public:
	ZMerger() = default;
	void async_merge(ZBuffer &z1, ZBuffer &z2);
};

void parallel_reduce_merge(std::vector<std::shared_ptr<ZMerger>> &mergers, std::vector<ZBuffer> &zbuffers, int active);

#endif//COURSE_RENDERER_ZMERGER_H
