//
// Created by dev on 7/8/22.
//

#ifndef COURSE_RENDERER_BPMLOADER_H
#define COURSE_RENDERER_BPMLOADER_H

// Isolated definition
// typedef struct { unsigned char x, y, z, w; } uchar4;

#include <vector_types.h>
void LoadBMPFile(uchar4 **dst, unsigned int *width,
                 unsigned int *height, const char *name);

#endif//COURSE_RENDERER_BPMLOADER_H
