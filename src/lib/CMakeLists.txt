cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

file(GLOB_RECURSE NVX_SRCS nvx_example/*.cpp nvx_kernels/*.cpp)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS} 
    ${TRT_INCLUDE_DIR}
    ${VisionWorks_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src/lib
)

link_directories(
    ${OpenCV_LIBS_DIR} 
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

add_library( nvx_demo_lib STATIC
    ${NVX_SRCS}
)

target_include_directories( nvx_demo_lib PRIVATE ${VisionWorks_INCLUDE_DIRS})

target_link_libraries( nvx_demo_lib
    nppc
    nppicc
    nppig
    cudart
    cudnn
    cublas
    vpi
    ${OpenCV_LIBS}
    ${VisionWorks_LIBRARIES}
    ${TRT_LIBRARY_INFER}
    ${TRT_LIBRARY_INFER_PLUGIN}
    ${TRT_LIBRARY_INFER_PARSER}
    ${TRT_LIBRARY_INFER_ONNX}
)

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