cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

file(GLOB_RECURSE CXX_SRCS *.cpp *.hpp)
file(GLOB CU_SRCS *.cu)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIR} 
    ${TRT_INCLUDE_DIR}
    ${VisionWorks_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src/lib
)

link_directories(
    ${OpenCV_LIBS_DIR} 
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

add_library( ${PROJECT_NAME}_lib STATIC
    ${CXX_SRCS}
    ${CU_SRCS} 
)

target_include_directories( ${PROJECT_NAME}_lib PRIVATE ${VisionWorks_INCLUDE_DIRS})

target_link_libraries( ${PROJECT_NAME}_lib
    nppc
    nppicc
    nppig
    cudart
    cudnn
    cublas
    ${OpenCV_LIBS}
    ${VisionWorks_LIBRARIES}
    ${TRT_LIBRARY_INFER}
    ${TRT_LIBRARY_INFER_PLUGIN}
    ${TRT_LIBRARY_INFER_PARSER}
    ${TRT_LIBRARY_INFER_ONNX}
)

# Compiling cuda sources with NVCC
set(CMAKE_CXX_FLAGS "-std=c++14 -Ofast -Wall -Wfatal-errors")
# SET(CUDA_SEPARABLE_COMPILATION ON) #  Required when compiling CUDA Kernels with Dyncamic Parallelism
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=compute_${CUDA_COMPUTE} -code=sm_${CUDA_COMPUTE}") # -rdc=true
CUDA_COMPILE(CU_OBJS ${CU_SRCS})