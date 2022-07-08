//
// Created by dev on 7/8/22.
//

#ifndef COURSE_RENDERER_KERNEL_CUH
#define COURSE_RENDERER_KERNEL_CUH

#include <helper_functions.h>
double main_cuda_launch(uint *dDest, int width, int height,
                           StopWatchInterface *timer);

#endif//COURSE_RENDERER_KERNEL_CUH
