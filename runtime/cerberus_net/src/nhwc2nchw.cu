// kernel to convert from OpenCV channel representation to channel-first
// see: https://docs.opencv.org/2.4/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html#how-the-image-matrix-is-stored-in-the-memory

const int BLOCK_SIZE = 1024;
#include <cuda_runtime.h>

__global__ void nhwc2nchwKernel(const unsigned char* __restrict__ source, float* __restrict__ dest,
    int channelSize, int channelsNum, int rowElems, int rowSize)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = idx / channelsNum;
    int channel = idx % channelsNum;

    // what would the row be if we didn't have any padding
    int row = idx / rowElems;
    int col = idx % rowElems;

    // actual element - skip padding
    int sourceIdx = row * rowSize + col;
    dest[channelSize * channel + offset] = (float) source[sourceIdx];
}

// we expect all memory to already reside on device so no need to allocate anything
void nhwc2nchw(const unsigned char * source, float * dest, int channelSize, int channelsNum, int rowElems, int rowSize, cudaStream_t Stream)
{
    const int nBlocks = (channelSize * channelsNum) / BLOCK_SIZE;
    nhwc2nchwKernel<<<nBlocks, BLOCK_SIZE, 0, Stream>>>(source, dest, channelSize, channelsNum, rowElems, rowSize);
}
