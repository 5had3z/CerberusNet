#include "correlation.cuh"

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
    return 0;
}

void CorrelationPlugin::terminate()
{
}

nvinfer1::Dims CorrelationPlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims)
{
    //output the result to channel
    return {0, 0, 0};
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
    return nvinfer1::DataType::kFLOAT;
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
__global__ void channels_first(const TensorAcc4R input, TensorAcc4R rinput,
	const int channels, const int height, const int width, const int pad_size)
{
	// n (batch size), c (num of channels), y (height), x (width)
	const int n = blockIdx.x;
	const int y = blockIdx.y;
	const int x = blockIdx.z;

	const int ch_off = threadIdx.x;

	for (int c = ch_off; c < channels; c += THREADS_PER_BLOCK) {
		rinput[n][y+pad_size][x+pad_size][c] = input[n][c][y][x];
	}
}

template <typename scalar_t>
__global__ void correlation_forward(TensorAcc4R output, const int nOutputChannels, const int outputHeight, const int outputWidth,
	const TensorAcc4R rInput1, const int nInputChannels, const int inputHeight, const int inputWidth, const TensorAcc4R rInput2,
	const int pad_size, const int kernel_size, const int max_displacement, const int stride1, const int stride2)
{
	// n (batch size), c (num of channels), y (height), x (width)
	const int n = blockIdx.x;
	const int y1 = blockIdx.y * stride1 + max_displacement;
	const int x1 = blockIdx.z * stride1 + max_displacement;
	const int c = threadIdx.x;

	__shared__ scalar_t prod_sum[THREADS_PER_BLOCK];

	// no significant speed-up in using chip memory for input1 sub-data, 
	// not enough chip memory size to accomodate memory per block for input2 sub-data
	// instead i've used device memory for both

	const int kernel_rad = (kernel_size - 1) / 2;
	const int displacement_rad = max_displacement / stride2;
	const int displacement_size = 2 * displacement_rad + 1;

	// Variables used for debugging boundary checks
	// const int pInputWidth = inputWidth + 2 * pad_size;
	// const int pInputHeight = inputHeight + 2 * pad_size;

	// element-wise product along channel axis
	for (int tj = -displacement_rad; tj <= displacement_rad; ++tj) {
		for (int ti = -displacement_rad; ti <= displacement_rad; ++ti) {
			prod_sum[c] = 0;
			const int x2 = x1 + ti*stride2;
			const int y2 = y1 + tj*stride2;

			for (int j = -kernel_rad; j <= kernel_rad; ++j) {
				for (int i = -kernel_rad; i <= kernel_rad; ++i) {
					for (int ch = c; ch < nInputChannels; ch += THREADS_PER_BLOCK) {
						// if (y1+j > pInputHeight || y2+j > pInputHeight || y1+j < 0 || y2+j < 0) {
						// 	printf("Height exceeded! 0 > ( %d | %d ) > %d\n", y1+j, y2+j, pInputHeight);
						// 	// continue;
						// }
						// if (x1+i > pInputWidth || x2+i > pInputWidth || x2+i < 0 || x1+i < 0) {
						// 	printf("Width exceeded! 0 > ( %d | %d ) > %d\n", x1+i, x2+i, pInputHeight);
						// 	// continue;
						// }
						prod_sum[c] += rInput1[n][y1+j][x1+i][ch] * rInput2[n][y2+j][x2+i][ch];
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
				const scalar_t nelems = kernel_size * kernel_size * nInputChannels;
				// if (tc > nOutputChannels) {
				// 	// This has not tripped any warnings yet
				// 	printf("Output Channels exceeded! 0 > %d > %d\n", tc, nOutputChannels);
				// 	continue;
				// }
				output[n][tc][blockIdx.y][blockIdx.z] = reduce_sum / nelems;
			}
		}
	}
}

int CorrelationPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
	const int nOutputChannels = output.size(1);
	const int outputHeight = output.size(2);
	const int outputWidth = output.size(3);

	const int nInputChannels = input1.size(1);
	const int inputHeight = input1.size(2);
	const int inputWidth = input1.size(3);

	cudaError_t err;
	const dim3 threadsPerBlock(THREADS_PER_BLOCK);
	dim3 blocks_grid(batchSize, inputHeight, inputWidth);

	const int paddedInputHeight = inputHeight + 2 * pad_size;
    const int paddedInputWidth  = inputWidth + 2 * pad_size;

	auto rInput1 = torch::zeros({batchSize, paddedInputHeight, paddedInputWidth, nInputChannels}, input1.options());
    auto rInput2 = torch::zeros({batchSize, paddedInputHeight, paddedInputWidth, nInputChannels}, input2.options());

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.scalar_type(), "channels_first_fwd_1",
		([&] {
			TensorAcc4R input1_acc  = input1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>();
			TensorAcc4R rInput1_acc = rInput1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>();

			channels_first<scalar_t> <<<blocks_grid, threadsPerBlock, 0, stream >>>(
				input1_acc, rInput1_acc, nInputChannels, inputHeight, inputWidth, pad_size);
		})
	);
	// check for errors
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error in input1 channels_first: %s\n", cudaGetErrorString(err));
		return err;
	}

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input2.scalar_type(), "channels_first_fwd_2",
		([&] {
			TensorAcc4R input2_acc  = input2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>();
			TensorAcc4R rInput2_acc = rInput2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>();

			channels_first<scalar_t> <<<blocks_grid, threadsPerBlock, 0, stream >>> (
				input2_acc, rInput2_acc, nInputChannels, inputHeight, inputWidth, pad_size);
		})
	);
	// check for errors
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error in input2 channels_first: %s\n", cudaGetErrorString(err));
		return err;
	}

	dim3 totalBlocksCorr(batchSize, outputHeight, outputWidth);

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.scalar_type(), "correlation_forward",
		([&] {
			TensorAcc4R output_acc  = output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>();
			TensorAcc4R rInput1_acc = rInput1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>();
			TensorAcc4R rInput2_acc = rInput2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>();

			correlation_forward<scalar_t> <<<totalBlocksCorr, threadsPerBlock, 0, stream >>> (
				output_acc, nOutputChannels, outputHeight, outputWidth,
				rInput1_acc, nInputChannels, inputHeight, inputWidth,
				rInput2_acc,
				pad_size, kernel_size, max_displacement, stride1, stride2);
		})
	);
	// check for errors
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error in correlation_forward_cuda_kernel: %s\n", cudaGetErrorString(err));
		return err;
	}

    return err;
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
