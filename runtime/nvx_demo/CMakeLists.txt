cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

file(GLOB_RECURSE NVX_SRCS src/*.cpp)

include_directories(
    ${OpenCV_INCLUDE_DIRS} 
    ${VisionWorks_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src
)

link_directories(
    ${OpenCV_LIBS_DIR} 
)

add_library( nvx_demo_lib STATIC
    ${NVX_SRCS}
)

target_include_directories( nvx_demo_lib PRIVATE ${VisionWorks_INCLUDE_DIRS})

target_link_libraries( nvx_demo_lib
    nppc
    nppicc
    nppig
    vpi
    ${OpenCV_LIBS}
    ${VisionWorks_LIBRARIES}
)
