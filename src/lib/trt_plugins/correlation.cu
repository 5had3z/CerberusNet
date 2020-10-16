#include "correlation.cuh"
#include "trt_utils.hpp"

#include <cassert>

#define CUDA_NUM_THREADS 1024
#define THREADS_PER_BLOCK 32

namespace
{
    const char* CORRELATION_PLUGIN_VERSION{"1"};
    const char* CORRELATION_PLUGIN_NAME{"Correlation_TRT"};
} // namespace

nvinfer1::PluginFieldCollection CorrelationPlugin::mFC{};
std::vector<nvinfer1::PluginField> CorrelationPlugin::mPluginAttributes;

CorrelationPlugin::CorrelationPlugin()
{
}

CorrelationPlugin::CorrelationPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char *>(data), *a = d;
    assert(d == a + length);
}

void CorrelationPlugin::serialize(void* buffer) const
{
    char* d = static_cast<char*>(buffer), *a = d;
    assert(d == a + getSerializationSize());
}

size_t CorrelationPlugin::getSerializationSize() const
{
    return 0;
}

int CorrelationPlugin::initialize()
{
	const int paddedInputHeight = m_inputH + 2 * pad_size;
    const int paddedInputWidth  = m_inputW + 2 * pad_size;
    const int tensor_volume = m_max_batch_size * paddedInputHeight * paddedInputWidth * m_inputC

    const size_t elem_size = 0;
    if (m_datatype == nvinfer1::DataType::kFLOAT) { elem_size = sizeof(float); }
    else if (m_datatype == nvinfer1::DataType::kHALF) { elem_size = sizeof(__half); }

    NV_CUDA_CHECK(cudaMalloc(&m_rInput1, tensor_volume * elem_size));
    NV_CUDA_CHECK(cudaMalloc(&m_rInput2, tensor_volume * elem_size));

    return 0;
}

void CorrelationPlugin::terminate()
{
}

nvinfer1::Dims CorrelationPlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims)
{
    //output the result to channel
    assert(index == 0 && nbInputDims == 2 && inputs[0].nbDims == 3 && inputs[1].nbDims == 3);
    assert(mNbInputChannels == inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2]);
    return nvinfer1::Dims3(m_outputC, m_outputH, m_outputW);
}

void CorrelationPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* CorrelationPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
nvinfer1::DataType CorrelationPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return m_datatype;
}

// Return true if output tensor is broadcast across a batch.
bool CorrelationPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool CorrelationPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

void CorrelationPlugin::configurePlugin(const nvinfer1::PluginTensorDesc* in, int nbInput, const nvinfer1::PluginTensorDesc* out, int nbOutput)
{
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void CorrelationPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void CorrelationPlugin::detachFromContext()
{
}

const char* CorrelationPlugin::getPluginType() const
{
    return CORRELATION_PLUGIN_NAME;
}

const char* CorrelationPlugin::getPluginVersion() const
{
    return CORRELATION_PLUGIN_VERSION;
}

void CorrelationPlugin::destroy()
{
    delete this;
}

// Clone the plugin
nvinfer1::IPluginV2IOExt* CorrelationPlugin::clone() const
{
    CorrelationPlugin *p = new CorrelationPlugin();
    p->setPluginNamespace(mPluginNamespace);
    return p;
}

template <typename scalar_t>
__global__ void channels_first(const scalar_t* __restrict__ input, scalar_t* rinput,
	const int channels, const int height, const int width, const int pad_size)
{
	// n (batch size), c (num of channels), y (height), x (width)
	const int n = blockIdx.x;
	const int y = blockIdx.y;
	const int x = blockIdx.z;

    const int ch_off = threadIdx.x;
    
    const int dimcyx = channels * height * width;
	const int dimyx = height * width;

	const int p_dimx = (width + 2 * pad_size);
	const int p_dimy = (height + 2 * pad_size);
	const int p_dimyxc = channels * p_dimy * p_dimx;
	const int p_dimxc = p_dimx * channels;

	for (int c = ch_off; c < channels; c += THREADS_PER_BLOCK) {
        rinput[n * p_dimyxc + (y + pad_size) * p_dimxc + (x + pad_size) * channels + c] =
            input[n * dimcyx + c * dimyx + y * width + x];
	}
}

template <typename scalar_t>
__global__ void correlation_forward(
    scalar_t* output, const int outputChannels, const int outputHeight, const int outputWidth,
    const scalar_t* __restrict__ rInput1, const int inputChannels, const int inputHeight, const int inputWidth,
    const scalar_t* __restrict__ rInput2,
	const int kernel_size, const int max_displacement, const int stride1, const int stride2)
{
    // n (batch size), c (num of channels), y (height), x (width)
	const int n = blockIdx.x;
	const int y1 = blockIdx.y * stride1 + max_displacement;
	const int x1 = blockIdx.z * stride1 + max_displacement;
    const int c = threadIdx.x;
    
    const int pdimyxc = inputHeight * inputWidth * inputChannels;
	const int pdimxc = inputWidth * inputChannels;
	const int pdimc = inputChannels;

	const int tdimcyx = nOutputChannels * outputHeight * outputWidth;
	const int tdimyx = outputHeight * outputWidth;
    const int tdimx = outputWidth;

	__shared__ scalar_t prod_sum[THREADS_PER_BLOCK];

	const int kernel_rad = (kernel_size - 1) / 2;
	const int displacement_rad = max_displacement / stride2;
	const int displacement_size = 2 * displacement_rad + 1;

	// element-wise product along channel axis
	for (int tj = -displacement_rad; tj <= displacement_rad; ++tj) {
		for (int ti = -displacement_rad; ti <= displacement_rad; ++ti) {
			prod_sum[c] = 0;
			const int x2 = x1 + ti*stride2;
			const int y2 = y1 + tj*stride2;

			for (int j = -kernel_rad; j <= kernel_rad; ++j) {
				for (int i = -kernel_rad; i <= kernel_rad; ++i) {
					for (int ch = c; ch < inputChannels; ch += THREADS_PER_BLOCK) {
                        const int indx1 = n * pdimyxc + (y1 + j) * pdimxc + (x1 + i) * pdimc + ch;
                        const int indx2 = n * pdimyxc + (y2 + j) * pdimxc + (x2 + i) * pdimc + ch;
                        
						prod_sum[c] += rInput1[indx1] * rInput2[indx2];
					}
				}
			}

			// accumulate 
			__syncthreads();
			if (c == 0) {
				scalar_t reduce_sum = 0;
				for (int index = 0; index < THREADS_PER_BLOCK; ++index) {
					reduce_sum += prod_sum[index];
				}
                const int tc = (tj + displacement_rad) * displacement_size + (ti + displacement_rad);
                const int tindx = n * tdimcyx + tc * tdimyx + blockIdx.y * tdimx + blockIdx.z;
				const scalar_t nelems = kernel_size * kernel_size * inputChannels;

				output[tindx] = reduce_sum / nelems;
			}
		}
	}
}

int CorrelationPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
	const dim3 threadsPerBlock(THREADS_PER_BLOCK);
	const dim3 reshape_grid(batchSize, m_inputH, m_inputW);
    const dim3 corr_grid(batchSize, m_outputH, m_outputW);
    
    const int pInputWidth = inputWidth + 2 * m_pad_size;
    const int pInputHeight = inputHeight + 2 * m_pad_size;

    if (m_datatype == nvinfer1::DataType::kFLOAT)
    {
        NV_CUDA_CHECK(channels_first<float> <<<reshape_grid, threadsPerBlock, 0, stream>>>(
            reinterpret_cast<const float*>(inputs[0]), reinterpret_cast<float*>(m_rInput1),
            m_inputC, m_inputH, m_inputW, m_pad_size));

        NV_CUDA_CHECK(channels_first<float> <<<reshape_grid, threadsPerBlock, 0, stream>>> (
            reinterpret_cast<const float*>(inputs[1]), reinterpret_cast<float*>(m_rInput2),
            m_inputC, m_inputH, m_inputW, m_pad_size));

        NV_CUDA_CHECK(correlation_forward<float> <<<corr_grid, threadsPerBlock, 0, stream>>> (
            reinterpret_cast<float*>(outputs[0]), m_outputC, m_outputH, m_outputW,
            reinterpret_cast<const float*>(m_rInput1), m_inputC, pInputHeight, pInputWidth,
            reinterpret_cast<const float*>(m_rInput2),
            m_kernel_size, m_max_displacement, m_stride1, m_stride2));
    }
    else if (m_datatype == nvinfer1::DataType::kHALF)
    {
        NV_CUDA_CHECK(channels_first<__half> <<<reshape_grid, threadsPerBlock, 0, stream>>>(
            reinterpret_cast<const __half*>(inputs[0]), reinterpret_cast<__half*>(m_rInput1),
            m_inputC, m_inputH, m_inputW, m_pad_size));

        NV_CUDA_CHECK(channels_first<__half> <<<reshape_grid, threadsPerBlock, 0, stream>>> (
            reinterpret_cast<const __half*>(inputs[1]), reinterpret_cast<__half*>(m_rInput2),
            m_inputC, m_inputH, m_inputW, m_pad_size));

        NV_CUDA_CHECK(correlation_forward<__half> <<<corr_grid, threadsPerBlock, 0, stream>>> (
            reinterpret_cast<__half*>(outputs[0]), m_outputC, m_outputH, m_outputW,
            reinterpret_cast<const __half*>(m_rInput1), m_inputC, pInputHeight, pInputWidth,
            reinterpret_cast<const __half*>(m_rInput2),
            m_kernel_size, m_max_displacement, m_stride1, m_stride2));
    }

    return cudaGetLastError();
}

CorrelationPlugin::CorrelationPlugin()
{
    mPluginAttributes.clear();

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* CorrelationPlugin::getPluginName() const
{
    return CORRELATION_PLUGIN_NAME;
}

const char* CorrelationPlugin::getPluginVersion() const
{
    return CORRELATION_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* CorrelationPlugin::getFieldNames()
{
    return &mFC;
}

nvinfer1::IPluginV2IOExt* CorrelationPlugin::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc)
{
    assert(!strcmp(name, getPluginName()));
    const nvinfer1::PluginField* fields = fc->fields;

    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
    }

    CorrelationPlugin* obj = new CorrelationPlugin();
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

nvinfer1::IPluginV2IOExt* CorrelationPlugin::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    CorrelationPlugin* obj = new CorrelationPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
