cmake_minimum_required(VERSION 3.20)

project(course_renderer CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(ENABLE_PRECOMPILED_HEADERS OFF)
set(CMAKE_NO_SYSTEM_FROM_IMPORTED ON)
set(CMAKE_CUDA_COMPILER /opt/cuda/bin/nvcc)
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math --dopt on -O3")
# set(CMAKE_CUDA_FLAGS_DEBUG  "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -g -rdynamic")

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS})
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
find_package(glm REQUIRED)

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
        src/render/misc/util.cu
        src/camera/camera.cu
        src/render/draw_caller/draw_caller.cpp
        src/render/profiling_timer/profiling_timer.cpp
        src/render/zbuffer/zbuffer.cu
        src/render/zbuffer/zfiller.cu
        src/render/draw_caller/rasterizer.cu
        src/render/image_resetter/image_resetter.cu
        src/render/zbuffer/zmerger.cu
        src/render/draw_caller/synchronizable.cpp
        src/scene/scene.cu
        src/render/culler/culler.cpp
        src/model/src/bounding_volume.cpp
        src/render/logger/logger.cpp
        src/scene/scene_loader.cpp 
        src/scene/predefined_scenes.cpp
        src/render/virtual_geometry/analyzer/mesh_analyzer.cu
        src/render/virtual_geometry/analyzer/mesh_analyzer_puppeteer.cpp
        src/render/virtual_geometry/manager/virtual_geometry_manager.cpp
        src/render/virtual_geometry/virtual_model/virtual_model.cpp
        src/render/virtual_geometry/object_manager/virtual_geometry_object_manager.cpp
        src/render/virtual_geometry/geometry_upsampler/geometry_upsampler.cu src/shader/shader_vgeom.cuh
        src/util/const.cpp src/util/implot.h src/util/implot.cpp src/util/implot_internal.h src/util/implot_items.cpp)
target_include_directories(course_renderer PRIVATE ./Common)

set_target_properties(course_renderer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        )

set(CMAKE_CXX_COMPILE_FEATURES cxx_constexpr)
target_link_libraries(${CMAKE_PROJECT_NAME})
target_link_libraries(${CMAKE_PROJECT_NAME}  ${OPENGL_LIBRARIES})
target_link_libraries(${CMAKE_PROJECT_NAME}  ${GLUT_LIBRARY})
target_link_libraries(${CMAKE_PROJECT_NAME} ${CUDA_LIBRARIES} -lcuda -L/usr/local/lib64)
target_link_libraries(${CMAKE_PROJECT_NAME} glm)

