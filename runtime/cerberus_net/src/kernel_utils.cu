// kernel to convert from OpenCV channel representation to channel-first
// see: https://docs.opencv.org/2.4/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html#how-the-image-matrix-is-stored-in-the-memory

const int BLOCK_SIZE = 1024;
#include <cuda_runtime.h>
#include <math_constants.h>
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
    if (offset < channel_stride)
    {
        scalar_t best_score = source[offset];
        intergral_t best_cls = 0;
        for (size_t cls=1; cls<n_classes; ++cls)
        {
            const scalar_t class_score = source[offset + cls*channel_stride];
            if (class_score > best_score)
            {
                best_score = class_score;
                best_cls = cls;
            }
        }
        output[offset] = best_cls;
    }
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

template<typename intergral_t, size_t n_classes>
__global__ void seg_image_Kernel(const intergral_t* __restrict__ argmax_image,
    u_char* __restrict__ rgb_image, const u_char* __restrict__ colour_map, size_t image_size)
{
    const int offset = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ u_char smem_colour_map[n_classes * 3U];
    if (threadIdx.x < n_classes * 3U)
    {
        smem_colour_map[threadIdx.x] = colour_map[threadIdx.x];
    }
    
    __syncthreads();

    if (offset < image_size)
    {
        const intergral_t class_id = argmax_image[offset];
        for (size_t ch=0U; ch<3U; ++ch)
        {
            rgb_image[3U * offset + ch] = smem_colour_map[3U * class_id + ch];
        }
    }
}

template<typename intergral_t, size_t n_classes>
void seg_image(const intergral_t* argmax_image, u_char* rgb_image, const u_char* colour_map,
    size_t image_size, cudaStream_t Stream)
{
    const int nBlocks = image_size / BLOCK_SIZE;
    seg_image_Kernel<intergral_t, n_classes><<<nBlocks, BLOCK_SIZE, 0, Stream>>>(
        argmax_image, rgb_image, colour_map, image_size);
}

template void seg_image<u_char, 19>(
    const u_char*, u_char*, const u_char*, size_t, cudaStream_t);

template<typename scalar_t>
__global__ void flow_image_Kernel(const scalar_t* __restrict__ flow_image,
    u_char* __restrict__ rgb_image, size_t image_size)
{
    const int offset = threadIdx.x + blockIdx.x * blockDim.x;
    const float scale_factor = 8.f;
    const float max_flow = 256.f;

    if (offset < image_size)
    {
        const scalar_t flow_x = flow_image[offset];
        const scalar_t flow_y = flow_image[offset + image_size];

        const scalar_t mag = sqrt(pow(flow_x, 2.f) + pow(flow_y, 2.f));

        const scalar_t h = atan2f(flow_y, flow_x) + CUDART_PI_F;
        const scalar_t s = min(max(mag * scale_factor / max_flow, 0.f), 1.f);
        const scalar_t v = min(max(scale_factor - s, 0.f), 1.f);

        const scalar_t C = v * s;
        const scalar_t X = C * (1.f - abs(fmodf(h / (CUDART_PI_F/3.f), 2) - 1.f));
        const scalar_t m = v - C;

        scalar_t r = 0;
        scalar_t g = 0;
        scalar_t b = 0;
        if(h >= 0.f && h < CUDART_PI_F/3.f){
            r = C,g = X,b = 0;
        }
        else if(h >= CUDART_PI_F/3.f && h < 2.f*CUDART_PI_F/3.f) {
            r = X,g = C,b = 0;
        }
        else if(h >= 2.f*CUDART_PI_F/3.f && h < CUDART_PI_F) {
            r = 0,g = C,b = X;
        }
        else if(h >= CUDART_PI_F && h < 4.f*CUDART_PI_F/3.f) {
            r = 0,g = X,b = C;
        }
        else if(h >= 4.f*CUDART_PI_F/3.f && h < 5.f*CUDART_PI_F/3.f) {
            r = X,g = 0,b = C;
        }
        else {
            r = C,g = 0,b = X;
        }

        rgb_image[3U * offset] = (r+m)*255.f;
        rgb_image[3U * offset + 1] = (g+m)*255.f;
        rgb_image[3U * offset + 2] = (b+m)*255.f;
    }
}

template<typename scalar_t>
void flow_image(const scalar_t* flow_image, u_char* rgb_image,
    size_t image_size, cudaStream_t Stream)
{
    const int nBlocks = image_size / BLOCK_SIZE;
    flow_image_Kernel<<<nBlocks, BLOCK_SIZE, 0, Stream>>>(
        flow_image, rgb_image, image_size);
}

template void flow_image<float>(const float*, u_char*, size_t, cudaStream_t);