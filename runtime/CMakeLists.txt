cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(monash_fyp LANGUAGES CXX VERSION 0.01)

# MANUALLY SET PATHS IF REQUIRED, example: "$ENV{HOME}/OpenCV-3.4.6"
# set(OPENCV_ROOT "")
# set(TRT_SDK_ROOT "")
set(CUDA_ARCH "6.1") #Check yours here: https://developer.nvidia.com/cuda-gpus
# add_definitions(-DOPENCV_CUDA) # Flag for CUDA Operations with OpenCV have been compiled
# add_definitions(-DUNIFIED) # Flag for unified memory

set(CMAKE_CXX_FLAGS "-std=c++17 -march=native -Ofast -Wall -Wfatal-errors")

# Cuda Architecture and Version Management
find_package(CUDA 10 REQUIRED cudart cublas curand)

string(REGEX REPLACE "\\." "" CUDA_COMPUTE "${CUDA_ARCH}")
set(GENCODES "${GENCODES} -gencode arch=compute_${CUDA_COMPUTE},code=compute_${CUDA_COMPUTE}")
if(CUDA_ARCH VERSION_GREATER "6.1")
   MESSAGE(STATUS "FP16 support detected")
   add_definitions(-DPRECISION="FP16")
else()
   MESSAGE(STATUS "Defaulting FP32")
   add_definitions(-DPRECISION="FP32")
endif()

# OpenCV Management
set(OpenCV_LIBS
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_highgui
    opencv_video
    opencv_videoio
    opencv_calib3d
    opencv_features2d)

# If OpenCV CUDA Support Has been Compiled
get_directory_property( DirDefs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS )
foreach( d ${DirDefs} )
    if(${d} MATCHES "OPENCV_CUDA")
      set(OpenCV_LIBS
        ${OpenCV_LIBS}
        opencv_cudev
        opencv_cudawarping
        opencv_cudaimgproc
        opencv_cudaarithm)
    endif()
endforeach()

if(NOT EXISTS ${OPENCV_ROOT})
    find_package(OpenCV REQUIRED COMPONENTS ${OpenCV_LIBS})
else()
    set(OpenCV_INCLUDE_DIR ${OPENCV_ROOT}/include)
    set(OpenCV_LIBS_DIR ${OPENCV_ROOT}/lib)
endif()

# TensorRT Management
find_path(TRT_INCLUDE_DIR NvInfer.h HINTS ${TRT_SDK_ROOT} PATH_SUFFIXES include)
if(${TRT_INCLUDE_DIR} MATCHES "TRT_INCLUDE_DIR-NOTFOUND")
    MESSAGE(FATAL_ERROR "-- Unable to find TensorRT headers. Please set path using -DTRT_SDK_ROOT")
endif()

find_library(TRT_LIBRARY_INFER nvinfer HINTS ${TRT_SDK_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
find_library(TRT_LIBRARY_INFER_PARSER nvparsers HINTS ${TRT_SDK_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
find_library(TRT_LIBRARY_INFER_ONNX nvonnxparser HINTS ${TRT_SDK_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
find_library(TRT_LIBRARY_INFER_PLUGIN nvinfer_plugin HINTS ${TRT_SDK_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
if((${TRT_LIBRARY_INFER} MATCHES "TRT_LIBRARY_INFER-NOTFOUND") OR (${TRT_LIBRARY_INFER_PLUGIN} MATCHES "TRT_LIBRARY_INFER_PLUGIN-NOTFOUND"))
    MESSAGE(FATAL_ERROR "-- Unable to find TensorRT libs. Please set path using -DTRT_SDK_ROOT")
else()
    set(TRT_LIBRARY ${TRT_LIBRARY_INFER} ${TRT_LIBRARY_INFER_PLUGIN} ${TRT_LIBRARY_INFER_PARSER} ${TRT_LIBRARY_INFER_ONNX})
    MESSAGE(STATUS "Found TensorRT libs at ${TRT_LIBRARY}")
endif()

#   VisionWorks, VPI and NPP Management
find_package(VisionWorks REQUIRED)
if (NOT ${VisionWorks_FOUND})
    MESSAGE(FATAL_ERROR "-- Unable to find VisionWorks. Make sure its been downloaded via nVIDIA SDK Manager and compiled")
endif()

find_package(CUDAToolkit REQUIRED nppc nppicc nppig)
if (NOT ${CUDAToolkit_FOUND})
    MESSAGE(FATAL_ERROR "-- Unable to find CUDAToolkit. Make sure you have CUDA installed")
endif()

find_package(vpi REQUIRED)
if (NOT ${vpi_FOUND})
    MESSAGE(FATAL_ERROR "-- Unable to find Vision Programming Interface (VPI). Make sure its been downloaded via nVIDIA SDK Manager")
endif()

# Add sub libs as subdirs
add_subdirectory(${PROJECT_SOURCE_DIR}/nvx_demo ${PROJECT_BINARY_DIR}/nvx_demo)
add_subdirectory(${PROJECT_SOURCE_DIR}/cerberus_net ${PROJECT_BINARY_DIR}/cerberus_demo)
