# Stereo-to-All

The aim of this project is to generate a pointcloud with motion vectors and semantic segmentation of stereoscopic image sequences. Targeting real-time deployment on an nVIDIA Xavier.

This is Public Repository for my Final Year Project since all my other projects/work is under the private Monash Motorsport git, so others don't have to take my word for it that I'm somewhat experienced and have some actual code to show publicly.

## Dependencies
### On the C++ Side of Things:
This project utlizes many libraries that are included with Jetpack 4.4 which include:

 - CUDA 10.2
 - TensorRT 7.1.0
 - cuDNN 8.0.0
 - VPI 0.2.0
 - OpenCV 4.1.1
 - CMake 3.8

### On the Python Side of Things:
The usual suspects for the most part:
 - Pytorch 1.5
 - Onnx 1.6.0
 - Numpy 1.18.3
 - Matplotlib