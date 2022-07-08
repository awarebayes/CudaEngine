//
// Created by dev on 7/8/22.
//

#ifndef COURSE_RENDERER_BILATERAL_KERNEL_CUH
#define COURSE_RENDERER_BILATERAL_KERNEL_CUH

#include <helper_functions.h>
void initTexture(int width, int height, uint *hImage);
void freeTextures();
void updateGaussian(float delta, int radius);
double bilateralFilterRGBA(uint *dDest, int width, int height,
                           float e_d, int radius, int iterations,
                           StopWatchInterface *timer);

#endif//COURSE_RENDERER_BILATERAL_KERNEL_CUH
