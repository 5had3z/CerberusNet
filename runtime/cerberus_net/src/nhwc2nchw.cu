// kernel to convert from OpenCV channel representation to channel-first
// see: https://docs.opencv.org/2.4/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html#how-the-image-matrix-is-stored-in-the-memory

const int BLOCK_SIZE = 1024;
#include <cuda_runtime.h>
#include <array>

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
    dest[channelSize * channel + offset] = (float) source[sourceIdx] / 255.f;
}

// we expect all memory to already reside on device so no need to allocate anything
void nhwc2nchw(const unsigned char * source, float * dest, int channelSize,
    int channelsNum, int rowElems, int rowSize, cudaStream_t Stream)
{
    const int nBlocks = (channelSize * channelsNum) / BLOCK_SIZE;
    nhwc2nchwKernel<<<nBlocks, BLOCK_SIZE, 0, Stream>>>(
        source, dest, channelSize, channelsNum, rowElems, rowSize);
}

template<typename scalar_t>
__global__ void normalizeChannelKernel(scalar_t* __restrict__ source,
    size_t channel_stride, scalar_t mean, scalar_t std)
{
    const int offset = threadIdx.x + blockIdx.x * blockDim.x;
    if (offset < channel_stride) { source[offset] = (source[offset] - mean) / std; }
}

template<typename scalar_t, size_t n_ch>
void normalize_image_chw(scalar_t* image, size_t ch_stride, const std::array<scalar_t, n_ch> &mean,
    const std::array<scalar_t, n_ch> &std, cudaStream_t Stream)
{
    const int nBlocks = ch_stride / BLOCK_SIZE;
    for (size_t ch=0; ch < n_ch; ++ch)
    {
        normalizeChannelKernel<scalar_t><<<nBlocks, BLOCK_SIZE, 0, Stream>>>(
            &image[ch*ch_stride], ch_stride, mean[ch], std[ch]);
    }
}

template void normalize_image_chw<float, 3ul>(float*, size_t, std::array<float, 3ul> const&,
    std::array<float, 3ul> const&, cudaStream_t);

template<typename scalar_t, typename intergral_t>
__global__ void argmax_chw_Kernel(const scalar_t* __restrict__ source,
    intergral_t* __restrict__ output, const size_t channel_stride, const size_t n_classes)
{
    const int offset = threadIdx.x + blockIdx.x * blockDim.x;
    scalar_t best_score = 0;
    intergral_t best_cls = n_classes+1;
    for (size_t cls=0; cls<n_classes; ++cls)
    {
        if (source[offset + cls*channel_stride] > best_score)
        {
            best_score = source[offset + cls*channel_stride];
            best_cls = cls;
        }
    }
    output[offset] = best_cls;
}

template<typename scalar_t, typename intergral_t>
void argmax_chw(const scalar_t* input, intergral_t* output,
    size_t n_classes, size_t ch_stride, cudaStream_t Stream)
{
    const int nBlocks = ch_stride / BLOCK_SIZE;
    argmax_chw_Kernel<scalar_t, intergral_t><<<nBlocks, BLOCK_SIZE, 0, Stream>>>(
        input, output, ch_stride, n_classes);
}

template void argmax_chw<float, unsigned char>(
    const float*, unsigned char*, size_t, size_t, cudaStream_t);