cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

file(GLOB_RECURSE TRT_PLUGINS_SRCS trt_plugins/*.cpp)
file(GLOB TRT_PLUGINS_CU_SRCS *.cu)

add_library( trt_plugins_lib STATIC
    ${TRT_PLUGINS_SRCS}
)

# Compiling cuda sources with NVCC
set(CMAKE_CXX_FLAGS "-std=c++14 -Ofast -Wall -Wfatal-errors")
# SET(CUDA_SEPARABLE_COMPILATION ON) #  Required when compiling CUDA Kernels with Dyncamic Parallelism
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=compute_${CUDA_COMPUTE} -code=sm_${CUDA_COMPUTE}") # -rdc=true
CUDA_COMPILE(CU_OBJS ${TRT_PLUGINS_CU_SRCS})