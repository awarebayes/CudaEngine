cmake_minimum_required(VERSION 3.20)

project(course_renderer CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS})
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(Common/imgui)


add_executable(course_renderer
        Common/imgui/imgui.cpp
        Common/imgui/imgui_draw.cpp
        Common/imgui/imgui_impl_opengl3.cpp
        Common/imgui/imgui_impl_glut.cpp
        Common/imgui/imgui_tables.cpp
        Common/imgui/imgui_demo.cpp
        Common/imgui/imgui_widgets.cpp

        src/main.cpp
        src/model/src/model.cpp
        src/model/src/pool.cpp
        src/util/stream_manager.cpp
        src/util/tiny_objloader.cpp
        src/model/src/texture.cu
        src/util/stb_image.cpp
        src/kernels/src/util.cu
        src/kernels/src/matrix.cu
        src/kernels/src/render.cu
        src/kernels/inc/shader.cuh src/kernels/inc/shader_impl.cuh
        src/camera/camera.cu
        src/render/draw_caller/draw_caller.cpp
        src/render/profiling_timer/profiling_timer.cpp
        src/render/zbuffer/zbuffer.cpp
        src/render/zbuffer/zfiller.cu
        src/render/draw_caller/rasterizer.cu
        src/render/image_resetter/image_resetter.cpp
        src/render/zbuffer/zmerger.cu
        src/render/draw_caller/synchronizable.cpp
        src/util/const.h src/render/scene/scene.cu src/render/scene/scene.h)
target_include_directories(course_renderer PRIVATE ./Common)

set_target_properties(course_renderer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        )

set(CMAKE_CXX_COMPILE_FEATURES cxx_constexpr)
target_link_libraries(${CMAKE_PROJECT_NAME})
target_link_libraries(${CMAKE_PROJECT_NAME}  ${OPENGL_LIBRARIES})
target_link_libraries(${CMAKE_PROJECT_NAME}  ${GLUT_LIBRARY})
target_link_libraries(${CMAKE_PROJECT_NAME} ${CUDA_LIBRARIES} -lcuda -L/usr/local/lib64)

