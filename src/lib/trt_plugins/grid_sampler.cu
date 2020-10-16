#include "grid_sampler.cuh"
#include "trt_utils.hpp"

#include <cassert>

namespace
{
    const char* GRID_SAMPLER_PLUGIN_VERSION{"1"};
    const char* GRID_SAMPLER_PLUGIN_NAME{"Grid_Sampler_TRT"};
} // namespace

nvinfer1::PluginFieldCollection GridSamplerPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> GridSamplerPluginCreator::mPluginAttributes;

GridSamplerPlugin::GridSamplerPlugin()
{
}

GridSamplerPlugin::GridSamplerPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char *>(data), *a = d;
    assert(d == a + length);
}

void GridSamplerPlugin::serialize(void* buffer) const
{
    char* d = static_cast<char*>(buffer), *a = d;
    assert(d == a + getSerializationSize());
}

size_t GridSamplerPlugin::getSerializationSize() const
{
    return 0;
}

int GridSamplerPlugin::initialize()
{
    return 0;
}

void GridSamplerPlugin::terminate()
{
}

nvinfer1::Dims GridSamplerPlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims)
{
    //output the result to channel
    return {0, 0, 0};
}

void GridSamplerPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* GridSamplerPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
nvinfer1::DataType GridSamplerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return nvinfer1::DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool GridSamplerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool GridSamplerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

void GridSamplerPlugin::configurePlugin(const nvinfer1::PluginTensorDesc* in, int nbInput, const nvinfer1::PluginTensorDesc* out, int nbOutput)
{
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void GridSamplerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void GridSamplerPlugin::detachFromContext()
{
}

const char* GridSamplerPlugin::getPluginType() const
{
    return GRID_SAMPLER_PLUGIN_NAME;
}

const char* GridSamplerPlugin::getPluginVersion() const
{
    return GRID_SAMPLER_PLUGIN_VERSION;
}

void GridSamplerPlugin::destroy()
{
    delete this;
}

// Clone the plugin
nvinfer1::IPluginV2IOExt* GridSamplerPlugin::clone() const
{
    GridSamplerPlugin *p = new GridSamplerPlugin();
    p->setPluginNamespace(mPluginNamespace);
    return p;
}

static inline bool within_bounds_2d(int64_t h, int64_t w, int64_t H, int64_t W) {
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
        return ((coord + 1.f) / 2) * (size - 1);
    } else {
        // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
        return ((coord + 1.f) * size - 1) / 2;
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
    GridSamplerPadding padding_mode, bool align_corners)
{
    coord = grid_sampler_unnormalize(coord, size, align_corners);
    if (padding_mode == GridSamplerPadding::Border) {
        // clip coordinates to image borders
        coord = clip_coordinates(coord, size);
    }
    else if (padding_mode == GridSamplerPadding::Reflection) {
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

template <typename scalar_t, typename index_t>
__global__ void grid_sampler_2d_kernel(const index_t nthreads,
    TensorInfo<scalar_t, index_t> input,
    TensorInfo<scalar_t, index_t> grid,
    TensorInfo<scalar_t, index_t> output,
    const Interpolation interpolation_mode, const Padding padding_mode, bool align_corners)
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
  
        if (interpolation_mode == Interpolation::Bilinear) {
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
        else if (interpolation_mode == Interpolation::Nearest) {
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

int GridSamplerPlugin::enqueue(int batchSize, const void* const* inputs,
    void** outputs, void* workspace, cudaStream_t stream)
{
    int64_t count = batchSize * m_input_h * m_input_w;

    grid_sampler_2d_kernel<scalar_t><<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count,
        reinterpret_cast<const float*>(inputs[0]), reinterpret_cast<const float*>(inputs[1]),
        reinterpret_cast<float*>(outputs[0]), m_interpolation_mode, m_padding_mode, m_align_corners);

    return cudaGetLastError();
}

GridSamplerPluginCreator::GridSamplerPluginCreator()
{
    mPluginAttributes.clear();

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GridSamplerPluginCreator::getPluginName() const
{
    return GRID_SAMPLER_PLUGIN_NAME;
}

const char* GridSamplerPluginCreator::getPluginVersion() const
{
    return GRID_SAMPLER_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* GridSamplerPluginCreator::getFieldNames()
{
    return &mFC;
}

nvinfer1::IPluginV2IOExt* GridSamplerPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc)
{
    assert(!strcmp(name, getPluginName()));
    const nvinfer1::PluginField* fields = fc->fields;

    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
    }

    GridSamplerPlugin* obj = new GridSamplerPlugin();
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

nvinfer1::IPluginV2IOExt* GridSamplerPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    GridSamplerPlugin* obj = new GridSamplerPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
