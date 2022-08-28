//
// Created by dev on 8/28/22.
//

#ifndef COURSE_RENDERER_SYNCHRONIZABLE_H
#define COURSE_RENDERER_SYNCHRONIZABLE_H

#include <driver_types.h>
class Synchronizable {
protected:
	cudaStream_t stream{};
public:
	Synchronizable();
	virtual ~Synchronizable();
	void await();
};

#endif//COURSE_RENDERER_SYNCHRONIZABLE_H
