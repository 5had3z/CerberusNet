#include "grid_sampler.hpp"
#include "trt_utils.hpp"
#include "cuda_fp16.h"

#include <limits>
#include <cassert>

// CUDA: grid stride looping
//
// int64_t _i_n_d_e_x specifically prevents overflow in the loop increment.
// If input.numel() < INT_MAX, _i_n_d_e_x < INT_MAX, except after the final
// iteration of the loop where _i_n_d_e_x += blockDim.x * gridDim.x can be
// greater than INT_MAX.  But in that case _i_n_d_e_x >= n, so there are no
// further iterations and the overflowed value in i=_i_n_d_e_x is not used.
#define CUDA_KERNEL_LOOP_TYPE(i, n, index_type)                         \
  int64_t _i_n_d_e_x = blockIdx.x * blockDim.x + threadIdx.x;           \
  for (index_type i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+=blockDim.x * gridDim.x, i=_i_n_d_e_x)

#define CUDA_KERNEL_LOOP(i, n) CUDA_KERNEL_LOOP_TYPE(i, n, int)

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int64_t N) {
  assert(N > 0 && "CUDA kernel launch blocks must be positive");
  constexpr int64_t max_int = std::numeric_limits<int>::max();

  // Round up division for positive number that cannot cause integer overflow
  auto block_num = (N - 1) / CUDA_NUM_THREADS + 1;
  assert(block_num <= max_int && "Can't schedule too many blocks on CUDA device");

  return static_cast<int>(block_num);
}

static __forceinline__ __device__
bool within_bounds_2d(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

// Unnormalizes a coordinate from the -1 to +1 scale to its pixel index value,
// where we view each pixel as an area between (idx - 0.5) and (idx + 0.5).
// if align_corners: -1 and +1 get sent to the centers of the corner pixels
//     -1 --> 0
//     +1 --> (size - 1)
//     scale_factor = (size - 1) / 2
// if not align_corners: -1 and +1 get sent to the image edges
//     -1 --> -0.5
//     +1 --> (size - 1) + 0.5 == size - 0.5
//     scale_factor = size / 2
template <typename scalar_t>
static __forceinline__ __device__
scalar_t grid_sampler_unnormalize(scalar_t coord, int size, bool align_corners)
{
    if (align_corners) {
        // unnormalize coord from [-1, 1] to [0, size - 1]
        return ((coord + static_cast<scalar_t>(1.f)) / static_cast<scalar_t>(2)) * static_cast<scalar_t>(size - 1);
    } else {
        // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
        return ((coord + static_cast<scalar_t>(1.f)) * static_cast<scalar_t>(size - 1)) / static_cast<scalar_t>(2);
    }
}

// Clips coordinates to between 0 and clip_limit - 1
template <typename scalar_t>
static __forceinline__ __device__
scalar_t clip_coordinates(scalar_t in, int clip_limit) {
    return ::min(static_cast<scalar_t>(clip_limit - 1), ::max(in, static_cast<scalar_t>(0)));
}

// Reflects coordinates until they fall between low and high (inclusive).
// The bounds are passed as twice their value so that half-integer values
// can be represented as ints.
template <typename scalar_t>
static __forceinline__ __device__
scalar_t reflect_coordinates(scalar_t in, int twice_low, int twice_high)
{
    if (twice_low == twice_high) { return static_cast<scalar_t>(0); }

    scalar_t min = static_cast<scalar_t>(twice_low) / 2;
    scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
    in = ::fabs(in - min);

    // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
    scalar_t extra = ::fmod(in, span);
    int flips = static_cast<int>(::floor(in / span));

    return flips % 2 == 0 ? extra + min : span - extra + min;
}

template<typename scalar_t> 
static __forceinline__ __device__ 
scalar_t safe_downgrade_to_int_range(scalar_t x){
    // -100.0 does not have special meaning. This is just to make sure 
    // it's not within_bounds_2d or within_bounds_3d, and does not cause 
    // undefined behavior. See #35506.  
    if (x > INT_MAX-1 || x < INT_MIN || !::isfinite(static_cast<double>(x))) 
        return static_cast<scalar_t>(-100.0); 
    return x;
}

// Computes the pixel source index value for a grid coordinate
template <typename scalar_t>
static __forceinline__ __device__
scalar_t grid_sampler_compute_source_index(scalar_t coord, int size,
    GridSampler::Padding padding_mode, bool align_corners)
{
    coord = grid_sampler_unnormalize(coord, size, align_corners);
    if (padding_mode == GridSampler::Padding::Border) {
        // clip coordinates to image borders
        coord = clip_coordinates(coord, size);
    }
    else if (padding_mode == GridSampler::Padding::Reflection) {
        // reflect coordinates by image borders
        if (align_corners) {
            coord = reflect_coordinates(coord, 0, 2*(size - 1));
        } else {
            coord = reflect_coordinates(coord, -1, 2*size - 1);
        }
        // clip coordinates to image borders
        coord = clip_coordinates(coord, size);
    }

    coord = safe_downgrade_to_int_range(coord); 
    return coord;
}

template <typename scalar_t>
__global__ void grid_sampler_kernel(const size_t nthreads,
    const scalar_t* __restrict__ input, size_t C, size_t inp_H, size_t inp_W,
    const scalar_t* __restrict__ grid,
    scalar_t* output, size_t out_H, size_t out_W,
    const GridSampler::Interpolation interpolation_mode,
    const GridSampler::Padding padding_mode, bool align_corners)
{
    // Input Strides
    size_t inp_sN = C * inp_H * inp_W;
    size_t inp_sC = inp_H * inp_W;
    size_t inp_sH = inp_W;
    size_t inp_sW = 1;

    // Grid Strides
    size_t grid_sN = inp_H * inp_W * 2;
    size_t grid_sH = inp_W * 2;
    size_t grid_sW = 2;
    size_t grid_sCoor = 1;

    // Output Strides
    size_t out_sN = C * out_H * out_W;
    size_t out_sC = out_H * out_W;
    size_t out_sH = out_W;
    size_t out_sW = 1;
  
    CUDA_KERNEL_LOOP_TYPE(index, nthreads, size_t) {
        const size_t w = index % out_W;
        const size_t h = (index / out_W) % out_H;
        const size_t n = index / (out_H * out_W);
        const size_t grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;
  
        // get the corresponding input x, y co-ordinates from grid
        scalar_t ix = grid[grid_offset];
        scalar_t iy = grid[grid_offset + grid_sCoor];
  
        ix = grid_sampler_compute_source_index(ix, inp_W, padding_mode, align_corners);
        iy = grid_sampler_compute_source_index(iy, inp_H, padding_mode, align_corners);
  
        if (interpolation_mode == Interpolation::Bilinear) {
            // get NE, NW, SE, SW pixel values from (x, y)
            size_t ix_nw = static_cast<size_t>(::floor(ix));
            size_t iy_nw = static_cast<size_t>(::floor(iy));
            size_t ix_ne = ix_nw + 1;
            size_t iy_ne = iy_nw;
            size_t ix_sw = ix_nw;
            size_t iy_sw = iy_nw + 1;
            size_t ix_se = ix_nw + 1;
            size_t iy_se = iy_nw + 1;
    
            // get surfaces to each neighbor:
            scalar_t nw = (ix_se - ix)    * (iy_se - iy);
            scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
            scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
            scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);
    
            // calculate bilinear weighted pixel value and set output pixel
            auto inp_ptr_NC = input + n * inp_sN;
            auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
            for (size_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
                *out_ptr_NCHW = static_cast<scalar_t>(0);
                if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
                    *out_ptr_NCHW += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
                }
                if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
                    *out_ptr_NCHW += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
                }
                if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
                    *out_ptr_NCHW += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
                }
                if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
                    *out_ptr_NCHW += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
                }
            }
        } 
        else if (interpolation_mode == Interpolation::Nearest) {
            size_t ix_nearest = static_cast<size_t>(::round(ix));
            size_t iy_nearest = static_cast<size_t>(::round(iy));
    
            // assign nearest neighor pixel value to output pixel
            auto inp_ptr_NC = input + n * inp_sN;
            auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
            for (size_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
                if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
                    *out_ptr_NCHW = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
                } else {
                    *out_ptr_NCHW = static_cast<scalar_t>(0);
                }
            }
        }
    }
}

int GridSamplerPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
	const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    const int64_t count = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[2] * inputDesc[0].dims.d[3];

    if (inputDesc[0].type == nvinfer1::DataType::kFLOAT)
    {
        grid_sampler_kernel<<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count,
            reinterpret_cast<const float*>(inputs[0]), inputDesc[0].dims.d[1], inputDesc[0].dims.d[2], inputDesc[0].dims.d[3],
            reinterpret_cast<const float*>(inputs[1]),
            reinterpret_cast<float*>(outputs[0]), outputDesc[0].dims.d[2], outputDesc[0].dims.d[3],
            m_interpolation_mode, m_padding_mode, m_align_corners);
    }
    // else if (inputDesc[0].type == nvinfer1::DataType::kHALF)
    // {
    //     grid_sampler_kernel<<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count,
    //         reinterpret_cast<const __half*>(inputs[0]), inputDesc[0].dims.d[1], inputDesc[0].dims.d[2], inputDesc[0].dims.d[3],
    //         reinterpret_cast<const __half*>(inputs[1]),
    //         reinterpret_cast<__half*>(outputs[0]), outputDesc[0].dims.d[2], outputDesc[0].dims.d[3],
    //         m_interpolation_mode, m_padding_mode, m_align_corners);
    // }

    return cudaGetLastError();
}
