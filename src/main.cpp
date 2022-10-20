/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


// OpenGL Graphics includes
#include <GL/freeglut.h>
#include <helper_gl.h>

// CUDA utilities and system includes
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <helper_cuda.h>// CUDA device initialization helper functions

// Shared Library Test Functions
#include <helper_functions.h>// CUDA SDK Helper functions

#include "camera/camera.h"
#include "helper_math.h"
#include "model/inc/pool.h"
#include "render/draw_caller/draw_caller.h"
#include "render/misc/draw_caller_args.cuh"
#include "scene/scene.h"
#include "scene/scene_loader.h"
#include "scene/predefined_scenes.h"

#ifdef NDEBUG
	#undef NDEBUG
	#include "imgui.h"
	#define NDEBUG
#else
	#include "imgui.h"
#endif

#include "imgui_impl_glut.h"
#include "imgui_impl_opengl3.h"

StopWatchInterface *timer = NULL;
unsigned int width, height;
unsigned int *hImage = NULL;

GLuint pbo;                                    // OpenGL pixel buffer object
struct cudaGraphicsResource *cuda_pbo_resource;// handles OpenGL-CUDA exchange
GLuint texid;                                  // texture
GLuint shader;


// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;// FPS count for averaging
int fpsLimit = 1;// FPS limit for sampling
const int REFRESH_DELAY = 10;

glm::vec3 offsetsss{0, 0, 0};


#define GL_TEXTURE_TYPE GL_TEXTURE_2D

void keyboard(unsigned char key, int /*x*/, int /*y*/) {
	switch (key) {
		case 27:
#if defined(__APPLE__) || defined(MACOSX)
			exit(EXIT_SUCCESS);
#else
			glutDestroyWindow(glutGetWindow());
			return;
#endif
			break;

		case 'a':
		case 'A':
			printf("> letter a is pressed! use me for input!");
			break;
		default:
			break;
	}

	glutPostRedisplay();
}

void timerEvent(int value) {
	if (glutGetWindow()) {
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	}
}

void reshape(int x, int y) {
	glViewport(0, 0, x, y);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}


GLuint compileASMShader(GLenum program_type, const char *code) {
	GLuint program_id;
	glGenProgramsARB(1, &program_id);
	glBindProgramARB(program_type, program_id);
	glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB,
	                   (GLsizei) strlen(code), (GLubyte *) code);

	GLint error_pos;
	glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

	if (error_pos != -1) {
		const GLubyte *error_string;
		error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
		printf("Program error at position: %d\n%s\n", (int) error_pos, error_string);
		return 0;
	}

	return program_id;
}

float computeFPS() {
	fpsCount++;

	if (fpsCount == fpsLimit) {
		char fps[256];
		float ifps = 1.0f / (sdkGetAverageTimerValue(&timer) / 1000.0f);
		sprintf(fps,
		        "CUDA Bresenham example with opengl! %3.f fps",
		        ifps);

		glutSetWindowTitle(fps);
		fpsCount = 0;
		fpsLimit = (int) MAX(ifps, 1.0f);

		sdkResetTimer(&timer);
		return ifps;
	}
	return 0;
}

// display results using OpenGL
void display() {
	sdkStartTimer(&timer);
	ImGuiIO &io = ImGui::GetIO();
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGLUT_NewFrame();

	// execute filter, writing results to pbo
	unsigned int *dResult;

	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, nullptr));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
	        (void **) &dResult, &num_bytes, cuda_pbo_resource));

	auto img = Image{(int)width, (int) height, dResult};
	auto mp = ModelPoolCreator().get();

	auto draw_caller = DrawCallerSigleton().get();
	auto scene = SceneSingleton().get();
	scene->tick();


	glViewport(0, 0, (GLsizei) io.DisplaySize.x, (GLsizei) io.DisplaySize.y);
	glClear(GL_COLOR_BUFFER_BIT);


	for (int i = 0; i < scene->get_n_models(); i++) {
		auto &model = scene->get_model(i);
		int posx = i / 100;
		int posy = i % 1000;
		// model.position.y = sin(glutGet(GLUT_ELAPSED_TIME) * 0.01 + posy * 0.01) * 0.4 + cos(glutGet(GLUT_ELAPSED_TIME) * 0.01 + posx * 0.01) * 0.2;
	}

	scene->display_menu();
	SceneLoaderSingleton().get()->display_widget();
	draw_caller->draw(scene->get_draw_call_args(), img);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, nullptr));

	// load texture from pbo
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBindTexture(GL_TEXTURE_2D, texid);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA,
	                GL_UNSIGNED_BYTE, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	// fragment program is required to display floating point texture
	glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, shader);
	glEnable(GL_FRAGMENT_PROGRAM_ARB);
	glDisable(GL_DEPTH_TEST);

	glBegin(GL_QUADS);
	{
		glTexCoord2f(0, 0);
		glVertex2f(-1, -1);
		glTexCoord2f(1, 0);
		glVertex2f(1, -1);
		glTexCoord2f(1, 1);
		glVertex2f(1, 1);
		glTexCoord2f(0, 1);
		glVertex2f(-1, 1);
	}
	glEnd();
	glBindTexture(GL_TEXTURE_TYPE, 0);
	glDisable(GL_FRAGMENT_PROGRAM_ARB);


	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	glutSwapBuffers();
	glutReportErrors();

	sdkStopTimer(&timer);

	computeFPS();
}

void initCuda() {
	// initialize gaussian mask
	sdkCreateTimer(&timer);
}

void cleanup() {
	sdkDeleteTimer(&timer);
	if (hImage) {
		free(hImage);
	}

	cudaGraphicsUnregisterResource(cuda_pbo_resource);

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGLUT_Shutdown();
	ImGui::DestroyContext();

	glDeleteBuffers(1, &pbo);
	glDeleteTextures(1, &texid);
	glDeleteProgramsARB(1, &shader);
}

// shader for displaying floating-point texture
static const char *shader_code =
        "!!ARBfp1.0\n"
        "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
        "END";

void initGLResources() {
	// create pixel buffer object
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(GLubyte) * 4,
	             hImage, GL_STREAM_DRAW_ARB);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(
	        &cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

	// create texture for display
	glGenTextures(1, &texid);
	glBindTexture(GL_TEXTURE_2D, texid);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA,
	             GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

	// load shader program
	shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

void initGL(int argc, char **argv) {
	// initialize GLUT
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(width, height);

	glutCreateWindow("CUDA Bresenham example");

	glutKeyboardFunc(keyboard);
	glutReshapeFunc(reshape);
	// glutIdleFunc(idle);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

	if (!isGLVersionSupported(2, 0) ||
	    !areGLExtensionsSupported(
	            "GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object")) {
		printf("Error: failed to get minimal extensions for demo\n");
		printf("This sample requires:\n");
		printf("  OpenGL version 2.0\n");
		printf("  GL_ARB_vertex_buffer_object\n");
		printf("  GL_ARB_pixel_buffer_object\n");
		exit(EXIT_FAILURE);
	}

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO &io = ImGui::GetIO();
	(void) io;

	io.DisplaySize.x = width;
	io.DisplaySize.y = height;

	ImGui::StyleColorsLight();
	assert(ImGui_ImplGLUT_Init());
	ImGui_ImplGLUT_InstallFuncs();

	assert(ImGui_ImplOpenGL3_Init("#version 330"));

	glutDisplayFunc(display);
}



void init_my_classes() {
	register_predefined_scenes();
	SceneLoaderSingleton().get()->load_scene("default");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
	// start logs

	printf("\nStarting...\n\n");

#if defined(__linux__)
	setenv("DISPLAY", ":0", 0);
#endif

	// load image to process
	hImage = static_cast<unsigned int *>(calloc(1920 * 1080 * 4, 1));
	height = 1080;
	width = 1920;


	// Default mode running with OpenGL visualization and in automatic mode
	// the output automatically changes animation
	printf("\n");

	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with
	// OpenGL/CUDA interop.
	initGL(argc, (char **) argv);


	initCuda();
	initGLResources();
	init_my_classes();

	glutCloseFunc(cleanup);

	printf("Running Standard Demonstration with GLUT loop...\n\n");
	// Main OpenGL loop that will run visualization for every vsync
	glutMainLoop();
}
