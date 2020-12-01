#include "correlation.hpp"
#include "trt_utils.hpp"
#include "cuda_fp16.h"

#define CUDA_NUM_THREADS 1024
#define THREADS_PER_BLOCK 32

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
__global__ void Correlation_Kernel(
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

	const int tdimcyx = outputChannels * outputHeight * outputWidth;
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
				const scalar_t nelems = kernel_size * kernel_size * pdimc;

				output[tindx] = reduce_sum / nelems;
			}
		}
	}
}

int CorrelationPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
	const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
	const dim3 threadsPerBlock(THREADS_PER_BLOCK);
	const dim3 reshape_grid(inputDesc[0].dims.d[0], inputDesc[0].dims.d[2], inputDesc[0].dims.d[3]);
    const dim3 corr_grid(inputDesc[0].dims.d[0], outputDesc[0].dims.d[2], outputDesc[0].dims.d[3]);
	
	const size_t pInputHeight = inputDesc[0].dims.d[2] + 2 * m_pad_size;
	const size_t pInputWidth = inputDesc[0].dims.d[3] + 2 * m_pad_size;
	const size_t pInputVolume = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1] * pInputHeight * pInputWidth;

	NV_CUDA_CHECK(cudaStreamSynchronize(stream));

	switch(inputDesc[0].type)
	{
		case nvinfer1::DataType::kFLOAT:
		{
			float* rInput1 = reinterpret_cast<float*>(workspace);
			float* rInput2 = reinterpret_cast<float*>(workspace) + pInputVolume * sizeof(float);
			NV_CUDA_CHECK(cudaMemsetAsync(rInput1, 0, pInputVolume * sizeof(float), stream_a));
			NV_CUDA_CHECK(cudaMemsetAsync(rInput2, 0, pInputVolume * sizeof(float), stream_b));

			channels_first<<<reshape_grid, threadsPerBlock, 0, stream_a>>> (
				reinterpret_cast<const float*>(inputs[0]), rInput1,
				inputDesc[0].dims.d[1], inputDesc[0].dims.d[2], inputDesc[0].dims.d[3], m_pad_size);

			channels_first<<<reshape_grid, threadsPerBlock, 0, stream_b>>> (
				reinterpret_cast<const float*>(inputs[1]), rInput2,
				inputDesc[1].dims.d[1], inputDesc[1].dims.d[2], inputDesc[1].dims.d[3], m_pad_size);

			NV_CUDA_CHECK(cudaStreamSynchronize(stream_a));
			NV_CUDA_CHECK(cudaStreamSynchronize(stream_b));

			Correlation_Kernel<<<corr_grid, threadsPerBlock, 0, stream>>> (
				reinterpret_cast<float*>(outputs[0]), outputDesc[0].dims.d[1], outputDesc[0].dims.d[2], outputDesc[0].dims.d[3],
				rInput1, inputDesc[0].dims.d[1], pInputHeight, pInputWidth, rInput2,
				m_kernel_size, m_max_displacement, m_stride1, m_stride2);
				
			break;
		}
		case nvinfer1::DataType::kHALF:
		{
			__half* rInput1 = reinterpret_cast<__half*>(workspace);
			__half* rInput2 = reinterpret_cast<__half*>(workspace) + pInputVolume * sizeof(__half);
			NV_CUDA_CHECK(cudaMemsetAsync(rInput1, 0, pInputVolume * sizeof(__half), stream_a));
			NV_CUDA_CHECK(cudaMemsetAsync(rInput2, 0, pInputVolume * sizeof(__half), stream_b));

			channels_first<<<reshape_grid, threadsPerBlock, 0, stream_a>>>(
				reinterpret_cast<const __half*>(inputs[0]), rInput1,
				inputDesc[0].dims.d[1], inputDesc[0].dims.d[2], inputDesc[0].dims.d[3], m_pad_size);

			channels_first<<<reshape_grid, threadsPerBlock, 0, stream_b>>> (
				reinterpret_cast<const __half*>(inputs[1]), rInput2,
				inputDesc[1].dims.d[1], inputDesc[1].dims.d[2], inputDesc[1].dims.d[3], m_pad_size);

			NV_CUDA_CHECK(cudaStreamSynchronize(stream_a));
			NV_CUDA_CHECK(cudaStreamSynchronize(stream_b));
			
			Correlation_Kernel<<<corr_grid, threadsPerBlock, 0, stream>>> (
				reinterpret_cast<__half*>(outputs[0]), outputDesc[0].dims.d[1], outputDesc[0].dims.d[2], outputDesc[0].dims.d[3],
				rInput1, inputDesc[0].dims.d[1], pInputHeight, pInputWidth, rInput2,
				m_kernel_size, m_max_displacement, m_stride1, m_stride2);

			break;
		}
		default:
		{
			std::cerr << "Correlation Plugin Unsupported Input Type";
			abort();
		}
	}

    return cudaGetLastError();
}
