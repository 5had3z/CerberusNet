#include "grid_sampler.cuh"

template <typename scalar_t, typename index_t>
__global__ void grid_sampler_2d_kernel(const index_t nthreads,
    TensorInfo<scalar_t, index_t> input, TensorInfo<scalar_t, index_t> grid, TensorInfo<scalar_t, index_t> output,
    const GridSamplerInterpolation interpolation_mode, const GridSamplerPadding padding_mode, bool align_corners)
{
    index_t C = input.sizes[1];
    index_t inp_H = input.sizes[2];
    index_t inp_W = input.sizes[3];
    index_t out_H = grid.sizes[1];
    index_t out_W = grid.sizes[2];
    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t inp_sH = input.strides[2];
    index_t inp_sW = input.strides[3];
    index_t grid_sN = grid.strides[0];
    index_t grid_sH = grid.strides[1];
    index_t grid_sW = grid.strides[2];
    index_t grid_sCoor = grid.strides[3];
    index_t out_sN = output.strides[0];
    index_t out_sC = output.strides[1];
    index_t out_sH = output.strides[2];
    index_t out_sW = output.strides[3];
  
    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
        const index_t w = index % out_W;
        const index_t h = (index / out_W) % out_H;
        const index_t n = index / (out_H * out_W);
        const index_t grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;
  
        // get the corresponding input x, y co-ordinates from grid
        scalar_t ix = grid.data[grid_offset];
        scalar_t iy = grid.data[grid_offset + grid_sCoor];
  
        ix = grid_sampler_compute_source_index(ix, inp_W, padding_mode, align_corners);
        iy = grid_sampler_compute_source_index(iy, inp_H, padding_mode, align_corners);
  
        if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
            // get NE, NW, SE, SW pixel values from (x, y)
            index_t ix_nw = static_cast<index_t>(::floor(ix));
            index_t iy_nw = static_cast<index_t>(::floor(iy));
            index_t ix_ne = ix_nw + 1;
            index_t iy_ne = iy_nw;
            index_t ix_sw = ix_nw;
            index_t iy_sw = iy_nw + 1;
            index_t ix_se = ix_nw + 1;
            index_t iy_se = iy_nw + 1;
    
            // get surfaces to each neighbor:
            scalar_t nw = (ix_se - ix)    * (iy_se - iy);
            scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
            scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
            scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);
    
            // calculate bilinear weighted pixel value and set output pixel
            auto inp_ptr_NC = input.data + n * inp_sN;
            auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
            for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
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
        else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
            index_t ix_nearest = static_cast<index_t>(::round(ix));
            index_t iy_nearest = static_cast<index_t>(::round(iy));
    
            // assign nearest neighor pixel value to output pixel
            auto inp_ptr_NC = input.data + n * inp_sN;
            auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
            for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
                if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
                    *out_ptr_NCHW = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
                } else {
                    *out_ptr_NCHW = static_cast<scalar_t>(0);
                }
            }
        }
    }
}

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
Tensor grid_sampler_2d_cuda(const Tensor& input, const Tensor& grid,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners)
{
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = grid.size(1);
    auto W = grid.size(2);
    auto output = at::empty({N, C, H, W}, input.options());
    int64_t count = N * H * W;

    if (count > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grid_sampler_2d_cuda", [&] {
            if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
                    canUse32BitIndexMath(output)) {
                grid_sampler_2d_kernel<scalar_t>
                    <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                        static_cast<int>(count),
                        getTensorInfo<scalar_t, int>(input),
                        getTensorInfo<scalar_t, int>(grid),
                        getTensorInfo<scalar_t, int>(output),
                        static_cast<GridSamplerInterpolation>(interpolation_mode),
                        static_cast<GridSamplerPadding>(padding_mode),
                        align_corners);
            } else {
                grid_sampler_2d_kernel<scalar_t>
                    <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                        count,
                        getTensorInfo<scalar_t, int64_t>(input),
                        getTensorInfo<scalar_t, int64_t>(grid),
                        getTensorInfo<scalar_t, int64_t>(output),
                        static_cast<GridSamplerInterpolation>(interpolation_mode),
                        static_cast<GridSamplerPadding>(padding_mode),
                        align_corners);
            }
        });
    }
    return output;
}