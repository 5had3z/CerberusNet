cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS} 
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/trt_plugins
)

link_directories(
    ${OpenCV_LIBS_DIR}
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

file(GLOB TRT_PLUGINS_SRCS trt_plugins/*.cpp)
file(GLOB TRT_PLUGINS_CU_SRCS trt_plugins/*.cu)

set(TMP_FLAGS ${CMAKE_CXX_FLAGS})
# Compiling cuda sources with NVCC
set(CMAKE_CXX_FLAGS "-std=c++14 -Og -Wall -Wfatal-errors")
# SET(CUDA_SEPARABLE_COMPILATION ON) #  Required when compiling CUDA Kernels with Dynamic Parallelism
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=compute_${CUDA_COMPUTE} -code=sm_${CUDA_COMPUTE}") # -rdc=true
CUDA_COMPILE(CU_PLUGIN_OBJS ${TRT_PLUGINS_CU_SRCS})
CUDA_COMPILE(CU_CERB_OBJS src/kernel_utils.cu)
set(CMAKE_CXX_FLAGS ${TMP_FLAGS})

add_library( trt_plugins_lib STATIC
    ${TRT_PLUGINS_SRCS}
    ${CU_PLUGIN_OBJS}
)

file(GLOB CERB_NET_SRCS src/*.cpp)

add_library( cerberus_lib STATIC
    ${CERB_NET_SRCS}
    ${CU_CERB_OBJS}
)

target_link_libraries( cerberus_lib
    trt_plugins_lib
    cudart
    stdc++fs
    ${TRT_LIBRARY_INFER}
    ${TRT_LIBRARY_INFER_PLUGIN}
    ${TRT_LIBRARY_INFER_PARSER}
    ${TRT_LIBRARY_INFER_ONNX}
)

add_executable (cerberus_run main.cpp)

target_link_libraries (
    cerberus_run
    cerberus_lib
    ${OpenCV_LIBS}
)
