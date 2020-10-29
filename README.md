# <s>Stereo</s>Mono-to-All

The aim of this project is to generate a pointcloud with motion vectors and semantic segmentation of <s>stereoscopic</s> monoscopic image sequences. Targeting real-time deployment on an nVIDIA Xavier.

This is Public Repository for my Final Year Project since all my other projects/work is under the private Monash Motorsport git, so others don't have to take my word for it that I'm somewhat experienced and have some actual code to show publicly.

If I get all 3 objectives working well, I call shotgun on the name Cerberus-Net.

Current WIP, still doing model exploration at the moment, no refinement or deeper cycles training yet.

## Cityscapes Segmentation + Flow + Depth
![Cityscapes HrnetV2 Segmentation + Flow + Depth](misc/E40_2.png)

## KITTI Segmentaiton + Flow
![KITTI HrnetV2 Segmentation + Flow](misc/E400_4.png)

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
 - Pytorch 1.6
 - Onnx 1.6.0
 - Numpy 1.18.3
 - Matplotlib
