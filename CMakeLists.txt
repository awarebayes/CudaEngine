cmake_minimum_required(VERSION 3.20)

project(course_renderer CXX CUDA)

set(CUDAToolkit_ROOT /opt/cuda)
set(CMAKE_CUDA_COMPILER /opt/cuda/bin/nvcc)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS})
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

file(GLOB renderer_SRC
        "*.cpp"
        "*.cu"
        )
add_executable(course_renderer ${renderer_SRC})
target_include_directories(course_renderer PRIVATE ./Common)

set_target_properties(course_renderer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(course_renderer ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
target_link_libraries(${CMAKE_PROJECT_NAME} ${CUDA_LIBRARIES} -lcuda)
