//
// Created by dev on 8/28/22.
//

#ifndef COURSE_RENDERER_CONST_H
#define COURSE_RENDERER_CONST_H

const int MAX_PIXELS_PER_KERNEL = 5000;

const int VIRTUAL_GEOMETRY_VERTICES = 1000;

extern int USE_THREADS;

int get_grid_size(int n);

int get_block_size(int n);

#endif//COURSE_RENDERER_CONST_H
