//
// Created by dev on 7/8/22.
//

#ifndef COURSE_RENDERER_BILATERALFILTER_CUH
#define COURSE_RENDERER_BILATERALFILTER_CUH

#include <GL/gl.h>

void varyEuclidean();
void computeFPS();
void display();
void keyboard(unsigned char key, int /*x*/, int /*y*/);
void timerEvent(int value);
void reshape(int x, int y);
void initCuda();
void cleanup();
GLuint compileASMShader(GLenum program_type, const char *code);
void initGLResources();
int runBenchmark(int argc, char **argv);
void initGL(int argc, char **argv);

#endif//COURSE_RENDERER_BILATERALFILTER_CUH
