#include "correlation_cuda_kernel_ARF.cuh"

#define CUDA_NUM_THREADS 1024
#define THREADS_PER_BLOCK 32

#include <ATen/Dispatch.h>

// using at::Half;

template <typename scalar_t>
__global__ void channels_first(const scalar_t* __restrict__ input, scalar_t* rinput, int channels, int height, int width, int pad_size)
{
	// n (batch size), c (num of channels), y (height), x (width)
	int n = blockIdx.x;
	int y = blockIdx.y;
	int x = blockIdx.z;

	int ch_off = threadIdx.x;
	scalar_t value;

	int dimcyx = channels * height * width;
	int dimyx = height * width;

	int p_dimx = (width + 2 * pad_size);
	int p_dimy = (height + 2 * pad_size);
	int p_dimyxc = channels * p_dimy * p_dimx;
	int p_dimxc = p_dimx * channels;

	for (int c = ch_off; c < channels; c += THREADS_PER_BLOCK) {
		value = input[n * dimcyx + c * dimyx + y * width + x];
		rinput[n * p_dimyxc + (y + pad_size) * p_dimxc + (x + pad_size) * channels + c] = value;
	}
}

template <typename scalar_t>
__global__ void correlation_forward(scalar_t* output, int nOutputChannels, int outputHeight, int outputWidth,
	const scalar_t* __restrict__ rInput1, int nInputChannels, int inputHeight, int inputWidth,
	const scalar_t* __restrict__ rInput2,
	int pad_size, int kernel_size, int max_displacement, int stride1, int stride2)
{
	// n (batch size), c (num of channels), y (height), x (width)

	int pInputWidth = inputWidth + 2 * pad_size;
	int pInputHeight = inputHeight + 2 * pad_size;

	int kernel_rad = (kernel_size - 1) / 2;
	int displacement_rad = max_displacement / stride2;
	int displacement_size = 2 * displacement_rad + 1;

	int n = blockIdx.x;
	int y1 = blockIdx.y * stride1 + max_displacement;
	int x1 = blockIdx.z * stride1 + max_displacement;
	int c = threadIdx.x;

	int pdimyxc = pInputHeight * pInputWidth * nInputChannels;
	int pdimxc = pInputWidth * nInputChannels;
	int pdimc = nInputChannels;

	int tdimcyx = nOutputChannels * outputHeight * outputWidth;
	int tdimyx = outputHeight * outputWidth;
	int tdimx = outputWidth;

	scalar_t nelems = kernel_size * kernel_size * pdimc;

	__shared__ scalar_t prod_sum[THREADS_PER_BLOCK];

	// no significant speed-up in using chip memory for input1 sub-data, 
	// not enough chip memory size to accomodate memory per block for input2 sub-data
	// instead i've used device memory for both 

	// element-wise product along channel axis
	for (int tj = -displacement_rad; tj <= displacement_rad; ++tj) {
		for (int ti = -displacement_rad; ti <= displacement_rad; ++ti) {
			prod_sum[c] = 0;
			int x2 = x1 + ti*stride2;
			int y2 = y1 + tj*stride2;

			for (int j = -kernel_rad; j <= kernel_rad; ++j) {
				for (int i = -kernel_rad; i <= kernel_rad; ++i) {
					for (int ch = c; ch < pdimc; ch += THREADS_PER_BLOCK) {
						int indx1 = n * pdimyxc + (y1 + j) * pdimxc + (x1 + i) * pdimc + ch;
						int indx2 = n * pdimyxc + (y2 + j) * pdimxc + (x2 + i) * pdimc + ch;

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
				int tc = (tj + displacement_rad) * displacement_size + (ti + displacement_rad);
				const int tindx = n * tdimcyx + tc * tdimyx + blockIdx.y * tdimx + blockIdx.z;
				output[tindx] = reduce_sum / nelems;
			}

		}
	}

}

template <typename scalar_t>
__global__ void correlation_backward_input1(int item, scalar_t* gradInput1, int nInputChannels, int inputHeight, int inputWidth,
	const scalar_t* __restrict__ gradOutput, int nOutputChannels, int outputHeight, int outputWidth,
	const scalar_t* __restrict__ rInput2,
	int pad_size, int kernel_size, int max_displacement, int stride1, int stride2)
{
	// n (batch size), c (num of channels), y (height), x (width)

	int n = item;
	int y = blockIdx.x * stride1 + pad_size;
	int x = blockIdx.y * stride1 + pad_size;
	int c = blockIdx.z;
	int tch_off = threadIdx.x;

	int kernel_rad = (kernel_size - 1) / 2;
	int displacement_rad = max_displacement / stride2;
	int displacement_size = 2 * displacement_rad + 1;

	int xmin = (x - kernel_rad - max_displacement) / stride1;
	int ymin = (y - kernel_rad - max_displacement) / stride1;

	int xmax = (x + kernel_rad - max_displacement) / stride1;
	int ymax = (y + kernel_rad - max_displacement) / stride1;

	if (xmax < 0 || ymax < 0 || xmin >= outputWidth || ymin >= outputHeight) {
		// assumes gradInput1 is pre-allocated and zero filled
		return;
	}

	if (xmin > xmax || ymin > ymax) {
		// assumes gradInput1 is pre-allocated and zero filled
		return;
	}

	xmin = max(0, xmin);
	xmax = min(outputWidth - 1, xmax);

	ymin = max(0, ymin);
	ymax = min(outputHeight - 1, ymax);

	int pInputWidth = inputWidth + 2 * pad_size;
	int pInputHeight = inputHeight + 2 * pad_size;

	int pdimyxc = pInputHeight * pInputWidth * nInputChannels;
	int pdimxc = pInputWidth * nInputChannels;
	int pdimc = nInputChannels;

	int tdimcyx = nOutputChannels * outputHeight * outputWidth;
	int tdimyx = outputHeight * outputWidth;
	int tdimx = outputWidth;

	int odimcyx = nInputChannels * inputHeight* inputWidth;
	int odimyx = inputHeight * inputWidth;
	int odimx = inputWidth;

	scalar_t nelems = kernel_size * kernel_size * nInputChannels;

	__shared__ scalar_t prod_sum[THREADS_PER_BLOCK];
	prod_sum[tch_off] = 0;

	for (int tc = tch_off; tc < nOutputChannels; tc += THREADS_PER_BLOCK) {

		int i2 = (tc % displacement_size - displacement_rad) * stride2;
		int j2 = (tc / displacement_size - displacement_rad) * stride2;

		int indx2 = n * pdimyxc + (y + j2)* pdimxc + (x + i2) * pdimc + c;

		scalar_t val2 = rInput2[indx2];

		for (int j = ymin; j <= ymax; ++j) {
			for (int i = xmin; i <= xmax; ++i) {
				int tindx = n * tdimcyx + tc * tdimyx + j * tdimx + i;
				prod_sum[tch_off] += gradOutput[tindx] * val2;
			}
		}
	}
	__syncthreads();

	if (tch_off == 0) {
		scalar_t reduce_sum = 0;
		for (int idx = 0; idx < THREADS_PER_BLOCK; idx++) {
			reduce_sum += prod_sum[idx];
		}
		const int indx1 = n * odimcyx + c * odimyx + (y - pad_size) * odimx + (x - pad_size);
		gradInput1[indx1] = reduce_sum / nelems;
	}

}

template <typename scalar_t>
__global__ void correlation_backward_input2(int item, scalar_t*  gradInput2, int nInputChannels, int inputHeight, int inputWidth,
	const scalar_t* __restrict__ gradOutput, int nOutputChannels, int outputHeight, int outputWidth,
	const scalar_t* __restrict__ rInput1,
	int pad_size, int kernel_size, int max_displacement, int stride1, int stride2)
{
	// n (batch size), c (num of channels), y (height), x (width)

	int n = item;
	int y = blockIdx.x * stride1 + pad_size;
	int x = blockIdx.y * stride1 + pad_size;
	int c = blockIdx.z;

	int tch_off = threadIdx.x;

	int kernel_rad = (kernel_size - 1) / 2;
	int displacement_rad = max_displacement / stride2;
	int displacement_size = 2 * displacement_rad + 1;

	int pInputWidth = inputWidth + 2 * pad_size;
	int pInputHeight = inputHeight + 2 * pad_size;

	int pdimyxc = pInputHeight * pInputWidth * nInputChannels;
	int pdimxc = pInputWidth * nInputChannels;
	int pdimc = nInputChannels;

	int tdimcyx = nOutputChannels * outputHeight * outputWidth;
	int tdimyx = outputHeight * outputWidth;
	int tdimx = outputWidth;

	int odimcyx = nInputChannels * inputHeight* inputWidth;
	int odimyx = inputHeight * inputWidth;
	int odimx = inputWidth;

	scalar_t nelems = kernel_size * kernel_size * nInputChannels;

	__shared__ scalar_t prod_sum[THREADS_PER_BLOCK];
	prod_sum[tch_off] = 0;

	for (int tc = tch_off; tc < nOutputChannels; tc += THREADS_PER_BLOCK) {
		int i2 = (tc % displacement_size - displacement_rad) * stride2;
		int j2 = (tc / displacement_size - displacement_rad) * stride2;

		int xmin = (x - kernel_rad - max_displacement - i2) / stride1;
		int ymin = (y - kernel_rad - max_displacement - j2) / stride1;

		int xmax = (x + kernel_rad - max_displacement - i2) / stride1;
		int ymax = (y + kernel_rad - max_displacement - j2) / stride1;

		if (xmax < 0 || ymax < 0 || xmin >= outputWidth || ymin >= outputHeight) {
			// assumes gradInput2 is pre-allocated and zero filled
			continue;
		}

		if (xmin > xmax || ymin > ymax) {
			// assumes gradInput2 is pre-allocated and zero filled
			continue;
		}

		xmin = max(0, xmin);
		xmax = min(outputWidth - 1, xmax);

		ymin = max(0, ymin);
		ymax = min(outputHeight - 1, ymax);

		int indx1 = n * pdimyxc + (y - j2)* pdimxc + (x - i2) * pdimc + c;
		scalar_t val1 = rInput1[indx1];

		for (int j = ymin; j <= ymax; ++j) {
			for (int i = xmin; i <= xmax; ++i) {
				int tindx = n * tdimcyx + tc * tdimyx + j * tdimx + i;
				prod_sum[tch_off] += gradOutput[tindx] * val1;
			}
		}
	}

	__syncthreads();

	if (tch_off == 0) {
		scalar_t reduce_sum = 0;
		for (int idx = 0; idx < THREADS_PER_BLOCK; idx++) {
			reduce_sum += prod_sum[idx];
		}
		const int indx2 = n * odimcyx + c * odimyx + (y - pad_size) * odimx + (x - pad_size);
		gradInput2[indx2] = reduce_sum / nelems;
	}

}

int correlation_forward_cuda_kernel(torch::Tensor& output, torch::Tensor& input1, 
	torch::Tensor& input2, torch::Tensor& rInput1, torch::Tensor& rInput2,
	int pad_size, int kernel_size, int max_displacement, int stride1,
	int stride2, int corr_type_multiply, cudaStream_t stream)
{
	int batchSize = output.size(0);
	int nOutputChannels = output.size(1);
	int outputHeight = output.size(2);
	int outputWidth = output.size(3);

	int nInputChannels = input1.size(1);
	int inputHeight = input1.size(2);
	int inputWidth = input1.size(3);

	const dim3 threadsPerBlock(THREADS_PER_BLOCK);

	dim3 blocks_grid(batchSize, inputHeight, inputWidth);

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.scalar_type(), "channels_first_fwd_1", ([&] {
		channels_first<scalar_t> <<<blocks_grid, threadsPerBlock, 0, stream >>>(
			input1.data_ptr<scalar_t>(), rInput1.data_ptr<scalar_t>(),
			nInputChannels, inputHeight, inputWidth, pad_size);
		})
	);

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input2.scalar_type(), "channels_first_fwd_2", ([&] {
		channels_first<scalar_t> <<<blocks_grid, threadsPerBlock, 0, stream >>> (
			input2.data_ptr<scalar_t>(), rInput2.data_ptr<scalar_t>(),
			nInputChannels, inputHeight, inputWidth, pad_size);
		})
	);

	dim3 totalBlocksCorr(batchSize, outputHeight, outputWidth);

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.scalar_type(), "correlation_forward", ([&] {
		correlation_forward<scalar_t> <<<totalBlocksCorr, threadsPerBlock, 0, stream >>> (
			output.data_ptr<scalar_t>(), nOutputChannels, outputHeight, outputWidth,
			rInput1.data_ptr<scalar_t>(), nInputChannels, inputHeight, inputWidth,
			rInput2.data_ptr<scalar_t>(),
			pad_size, kernel_size, max_displacement, stride1, stride2);
		})
	);

	// check for errors
	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error in correlation_forward_cuda_kernel: %s\n", cudaGetErrorString(err));
		return 0;
	}

	return 1;
}

int correlation_backward_cuda_kernel( torch::Tensor& gradOutput,
	torch::Tensor& input1, torch::Tensor& input2, torch::Tensor& gradInput1,
	torch::Tensor& gradInput2, torch::Tensor& rInput1, torch::Tensor& rInput2,
	int pad_size, int kernel_size, int max_displacement, int stride1,
	int stride2, int corr_type_multiply, cudaStream_t stream)
{
	cudaError_t err;
	int batchSize		= gradOutput.size(0);
	int nOutputChannels = gradOutput.size(1);
	int outputHeight	= gradOutput.size(2);
	int outputWidth		= gradOutput.size(3);

	int nInputChannels  = input1.size(1);
	int inputHeight 	= input1.size(2);
	int inputWidth 		= input1.size(3);

	dim3 blocks_grid(batchSize, inputHeight, inputWidth);
	const dim3 threadsPerBlock(THREADS_PER_BLOCK);

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.scalar_type(), "channels_first_bck_1", ([&] {
		channels_first<scalar_t> <<<blocks_grid, threadsPerBlock, 0, stream >>>(
			input1.data_ptr<scalar_t>(), rInput1.data_ptr<scalar_t>(),
			nInputChannels, inputHeight, inputWidth, pad_size );
		})
	);
	// check for errors
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error in input1 channels_first: %s\n", cudaGetErrorString(err));
		return 0;
	}

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input2.scalar_type(), "channels_first_bck_2", ([&] {
		channels_first<scalar_t> <<<blocks_grid, threadsPerBlock, 0, stream >>>(
			input2.data_ptr<scalar_t>(), rInput2.data_ptr<scalar_t>(),
			nInputChannels, inputHeight, inputWidth, pad_size );
		})
	);
	// check for errors
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error in input2 channels_first: %s\n", cudaGetErrorString(err));
		return 0;
	}

	std::cout << "rInput1 batch: " << rInput1.size(0) << " ch: " << rInput1.size(1) <<
		" h: " << rInput1.size(2) << " w: " << rInput1.size(3) << std::endl;

	std::cout << "gradInput1 batch: " << gradInput1.size(0) << " ch: " << gradInput1.size(1) <<
		" h: " << gradInput1.size(2) << " w: " << gradInput1.size(3) << std::endl;

	std::cout << "rInput2 batch: " << rInput2.size(0) << " ch: " << rInput2.size(1) <<
		" h: " << rInput2.size(2) << " w: " << rInput2.size(3) << std::endl;

	std::cout << "gradInput2 batch: " << gradInput2.size(0) << " ch: " << gradInput2.size(1) <<
		" h: " << gradInput2.size(2) << " w: " << gradInput2.size(3) << std::endl;

	std::cout << "gradOutput batch: " << gradOutput.size(0) << " ch: " << gradOutput.size(1) <<
		" h: " << gradOutput.size(2) << " w: " << gradOutput.size(3) << std::endl;

	std::cout << "gradOutput strides batch: " << gradOutput.stride(0) << " ch: " << gradOutput.stride(1) <<
		" h: " << gradOutput.stride(2) << " w: " << gradOutput.stride(3) << std::endl;

	dim3 totalBlocksCorr(inputHeight, inputWidth, nInputChannels);

	for (int n = 0; n < batchSize; ++n) {
		AT_DISPATCH_FLOATING_TYPES_AND_HALF(input2.scalar_type(), "correlation_backward_input1", ([&] {
			correlation_backward_input1<scalar_t> <<<totalBlocksCorr, threadsPerBlock, 0, stream >>> (
				n, gradInput1.data_ptr<scalar_t>(), nInputChannels, inputHeight, inputWidth,
				gradOutput.data_ptr<scalar_t>(), nOutputChannels, outputHeight, outputWidth,
				rInput2.data_ptr<scalar_t>(), pad_size, kernel_size, max_displacement, stride1, stride2);
			})
		);
		// check for errors
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("error in correlation_backward_input1[iter:%d]: %s\n", n, cudaGetErrorString(err));
			return 0;
		}
	}

	for (int n = 0; n < batchSize; n++) {
		AT_DISPATCH_FLOATING_TYPES_AND_HALF(rInput1.scalar_type(), "correlation_backward_input2", ([&] {
			correlation_backward_input2<scalar_t> <<<totalBlocksCorr, threadsPerBlock, 0, stream >>>(
				n, gradInput2.data_ptr<scalar_t>(), nInputChannels, inputHeight, inputWidth,
				gradOutput.data_ptr<scalar_t>(), nOutputChannels, outputHeight, outputWidth,
				rInput1.data_ptr<scalar_t>(), pad_size, kernel_size, max_displacement, stride1, stride2);
			})
		);
		// check for errors
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("error in correlation_backward_input2[iter:%d]: %s\n", n, cudaGetErrorString(err));
			return 0;
		}
	}

	return 1;
}
