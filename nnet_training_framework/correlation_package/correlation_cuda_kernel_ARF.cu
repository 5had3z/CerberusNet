#include "correlation_cuda_kernel_ARF.cuh"

#define CUDA_NUM_THREADS 1024
#define THREADS_PER_BLOCK 32

#include <ATen/Dispatch.h>

// using at::Half;
#define TensorAcc4R torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits>

template <typename scalar_t>
__global__ void channels_first(const TensorAcc4R input, TensorAcc4R rinput, int channels, int height, int width, int pad_size)
{
	// n (batch size), c (num of channels), y (height), x (width)
	int n = blockIdx.x;
	int y = blockIdx.y;
	int x = blockIdx.z;

	int ch_off = threadIdx.x;

	for (int c = ch_off; c < channels; c += THREADS_PER_BLOCK) {
		rinput[n][y+pad_size][x+pad_size][c] = input[n][c][y][x];
	}
}

template <typename scalar_t>
__global__ void correlation_forward(TensorAcc4R output, int nOutputChannels, int outputHeight, int outputWidth,
	const TensorAcc4R rInput1, int nInputChannels, int inputHeight, int inputWidth, const TensorAcc4R rInput2,
	int pad_size, int kernel_size, int max_displacement, int stride1, int stride2)
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

	const int pInputWidth = inputWidth + 2 * pad_size;
	const int pInputHeight = inputHeight + 2 * pad_size;

	// element-wise product along channel axis
	for (int tj = -displacement_rad; tj <= displacement_rad; ++tj) {
		for (int ti = -displacement_rad; ti <= displacement_rad; ++ti) {
			prod_sum[c] = 0;
			const int x2 = x1 + ti*stride2;
			const int y2 = y1 + tj*stride2;

			for (int j = -kernel_rad; j <= kernel_rad; ++j) {
				for (int i = -kernel_rad; i <= kernel_rad; ++i) {
					for (int ch = c; ch < nInputChannels; ch += THREADS_PER_BLOCK) {
						if (y1+j > pInputHeight || y2+j > pInputHeight || y1+j < 0 || y2+j < 0) {
							// printf("Height exceeded! 0 > ( %d | %d ) > %d\n", y1+j, y2+j, pInputHeight);
							continue;
						}
						if (x1+i > pInputWidth || x2+i > pInputWidth || x2+i < 0 || x1+i < 0) {
							// printf("Width exceeded! 0 > ( %d | %d ) > %d\n", x1+i, x2+i, pInputHeight);
							continue;
						}
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
				if (tc > nOutputChannels) {
					// This has not tripped any warnings yet
					// printf("Output Channels exceeded! 0 > %d > %d\n", tc, nOutputChannels);
					continue;
				}
				output[n][tc][blockIdx.y][blockIdx.z] = reduce_sum / nelems;
			}
		}
	}
}

template <typename scalar_t>
__global__ void correlation_backward_input1(int item, TensorAcc4R gradInput1, int nInputChannels, int inputHeight, int inputWidth,
	const TensorAcc4R gradOutput, int nOutputChannels, int outputHeight, int outputWidth,
	const TensorAcc4R rInput2,
	int pad_size, int kernel_size, int max_displacement, int stride1, int stride2)
{
	// n (batch size), c (num of channels), y (height), x (width)

	const int n = item;
	const int y = blockIdx.x * stride1 + pad_size;
	const int x = blockIdx.y * stride1 + pad_size;
	const int c = blockIdx.z;
	const int tch_off = threadIdx.x;

	const int kernel_rad = (kernel_size - 1) / 2;
	const int displacement_rad = max_displacement / stride2;
	const int displacement_size = 2 * displacement_rad + 1;

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

	scalar_t nelems = kernel_size * kernel_size * nInputChannels;

	__shared__ scalar_t prod_sum[THREADS_PER_BLOCK];
	prod_sum[tch_off] = 0;

	for (int tc = tch_off; tc < nOutputChannels; tc += THREADS_PER_BLOCK) {

		const int i2 = (tc % displacement_size - displacement_rad) * stride2;
		const int j2 = (tc / displacement_size - displacement_rad) * stride2;

		const int pInputWidth = inputWidth+pad_size*2;
		const int pInputHeight = inputHeight+pad_size*2;
		if (x+i2 > pInputWidth || y+j2 > pInputHeight || x+i2 < 0 || y+j2 < 0) {
			// printf("Input Width/Height (%d,%d) exceeded! (%d,%d)\n",
			// 		pInputWidth, pInputHeight, y+j2, x+i2);
			continue;
		}
		const scalar_t val2 = rInput2[n][y+j2][x+i2][c];

		for (int j = ymin; j <= ymax; ++j) {
			for (int i = xmin; i <= xmax; ++i) {
				prod_sum[tch_off] += gradOutput[n][tc][j][i] * val2;
			}
		}
	}
	__syncthreads();

	if (tch_off == 0) {
		scalar_t reduce_sum = 0;
		for (int idx = 0; idx < THREADS_PER_BLOCK; idx++) {
			reduce_sum += prod_sum[idx];
		}
		gradInput1[n][c][y-pad_size][x-pad_size] = reduce_sum / nelems;
	}

}

template <typename scalar_t>
__global__ void correlation_backward_input2(int item, TensorAcc4R gradInput2, int nInputChannels, int inputHeight, int inputWidth,
	const TensorAcc4R gradOutput, int nOutputChannels, int outputHeight, int outputWidth, const TensorAcc4R rInput1,
	int pad_size, int kernel_size, int max_displacement, int stride1, int stride2)
{
	// n (batch size), c (num of channels), y (height), x (width)

	const int n = item;
	const int y = blockIdx.x * stride1 + pad_size;
	const int x = blockIdx.y * stride1 + pad_size;
	const int c = blockIdx.z;

	const int tch_off = threadIdx.x;

	const int kernel_rad = (kernel_size - 1) / 2;
	const int displacement_rad = max_displacement / stride2;
	const int displacement_size = 2 * displacement_rad + 1;

	scalar_t nelems = kernel_size * kernel_size * nInputChannels;

	__shared__ scalar_t prod_sum[THREADS_PER_BLOCK];
	prod_sum[tch_off] = 0;

	for (int tc = tch_off; tc < nOutputChannels; tc += THREADS_PER_BLOCK) {
		const int i2 = (tc % displacement_size - displacement_rad) * stride2;
		const int j2 = (tc / displacement_size - displacement_rad) * stride2;

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

		const scalar_t val1 = rInput1[n][y-j2][x-i2][c];

		for (int j = ymin; j <= ymax; ++j) {
			for (int i = xmin; i <= xmax; ++i) {
				prod_sum[tch_off] += gradOutput[n][tc][j][i] * val1;
			}
		}
	}

	__syncthreads();

	if (tch_off == 0) {
		scalar_t reduce_sum = 0;
		for (int idx = 0; idx < THREADS_PER_BLOCK; idx++) {
			reduce_sum += prod_sum[idx];
		}
		gradInput2[n][c][y-pad_size][x-pad_size] = reduce_sum / nelems;
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

	cudaError_t err;
	const dim3 threadsPerBlock(THREADS_PER_BLOCK);
	dim3 blocks_grid(batchSize, inputHeight, inputWidth);

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
		return 0;
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
		return 0;
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
	const int batchSize			= gradOutput.size(0);
	const int nOutputChannels 	= gradOutput.size(1);
	const int outputHeight		= gradOutput.size(2);
	const int outputWidth		= gradOutput.size(3);

	const int nInputChannels	= input1.size(1);
	const int inputHeight 		= input1.size(2);
	const int inputWidth 		= input1.size(3);

	dim3 blocks_grid(batchSize, inputHeight, inputWidth);
	const dim3 threadsPerBlock(THREADS_PER_BLOCK);

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.scalar_type(), "channels_first_bck_1", ([&]
		{
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
		return 0;
	}

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input2.scalar_type(), "channels_first_bck_2", ([&]
		{
			TensorAcc4R input2_acc  = input2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>();
			TensorAcc4R rInput2_acc = rInput2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>();

			channels_first<scalar_t> <<<blocks_grid, threadsPerBlock, 0, stream >>>(
				input2_acc, rInput2_acc, nInputChannels, inputHeight, inputWidth, pad_size);
		})
	);
	// check for errors
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error in input2 channels_first: %s\n", cudaGetErrorString(err));
		return 0;
	}

	dim3 totalBlocksCorr(inputHeight, inputWidth, nInputChannels);

	for (int n = 0; n < batchSize; ++n) {
		AT_DISPATCH_FLOATING_TYPES_AND_HALF(input2.scalar_type(), "correlation_backward_input1",
			([&] {
				TensorAcc4R gradInput1_acc  = gradInput1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>();
				TensorAcc4R gradOutput_acc  = gradOutput.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>();
				TensorAcc4R rInput2_acc  = rInput2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>();

				correlation_backward_input1<scalar_t> <<<totalBlocksCorr, threadsPerBlock, 0, stream >>> (
					n, gradInput1_acc, nInputChannels, inputHeight, inputWidth,
					gradOutput_acc, nOutputChannels, outputHeight, outputWidth,
					rInput2_acc, pad_size, kernel_size, max_displacement, stride1, stride2);
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
		AT_DISPATCH_FLOATING_TYPES_AND_HALF(rInput1.scalar_type(), "correlation_backward_input2",
			([&] {
				TensorAcc4R gradInput2_acc  = gradInput2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>();
				TensorAcc4R gradOutput_acc  = gradOutput.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>();
				TensorAcc4R rInput1_acc  = rInput1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>();

				correlation_backward_input2<scalar_t> <<<totalBlocksCorr, threadsPerBlock, 0, stream >>>(
					n, gradInput2_acc, nInputChannels, inputHeight, inputWidth,
					gradOutput_acc, nOutputChannels, outputHeight, outputWidth,
					rInput1_acc, pad_size, kernel_size, max_displacement, stride1, stride2);
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
