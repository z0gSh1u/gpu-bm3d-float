cmake_minimum_required(VERSION 3.10)
project(bm3d)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories("D:/gpu-bm3d/")

set(SOURCE_FILES bm3d.cu denoise.cpp getopt.cpp)

cuda_add_executable(bm3d ${SOURCE_FILES} getopt.h Cimg.h params.h stopwatch.hpp bm3d.h)

target_link_libraries(bm3d ${CUDA_LIBRARIES} CUDA::cufft)