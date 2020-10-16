#include "correlation.cuh"

#define CUDA_NUM_THREADS 1024
#define THREADS_PER_BLOCK 32

// using at::Half;
#define TensorAcc4R torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits>

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

int correlation_forward_cuda_kernel(torch::Tensor& output,
	const torch::Tensor& input1, const torch::Tensor& input2,
	const int pad_size, const int kernel_size, const int max_displacement, const int stride1,
	const int stride2, const int corr_type_multiply)
{
	const int batchSize = output.size(0);
	const int nOutputChannels = output.size(1);
	const int outputHeight = output.size(2);
	const int outputWidth = output.size(3);

	const int nInputChannels = input1.size(1);
	const int inputHeight = input1.size(2);
	const int inputWidth = input1.size(3);

	cudaError_t err;
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
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

torch::Tensor correlation_forward_cuda(
    const torch::Tensor& input1, const torch::Tensor& input2, int pad_size,
    int kernel_size, int max_displacement, int stride1, int stride2, int corr_type_multiply)
{
    const int batchSize         = input1.size(0);
    const int nInputChannels    = input1.size(1);
    const int inputHeight       = input1.size(2);
    const int inputWidth        = input1.size(3);

    const int kernel_radius = (kernel_size - 1) / 2;
    const int border_radius = kernel_radius + max_displacement;

    const int paddedInputHeight = inputHeight + 2 * pad_size;
    const int paddedInputWidth  = inputWidth + 2 * pad_size;

    const int nOutputChannels   = ((max_displacement/stride2)*2 + 1) * ((max_displacement/stride2)*2 + 1);
    const int outputHeight      = ceil(static_cast<float>(paddedInputHeight - 2 * border_radius) / static_cast<float>(stride1));
    const int outputwidth       = ceil(static_cast<float>(paddedInputWidth - 2 * border_radius) / static_cast<float>(stride1));

    auto output = torch::zeros({batchSize, nOutputChannels, outputHeight, outputwidth}, input1.options());

    const int success = correlation_forward_cuda_kernel(
        output, input1, input2, pad_size, kernel_size,
        max_displacement, stride1, stride2, corr_type_multiply);

    //check for errors
    if (!success) { AT_ERROR("CUDA correlation_forward_cuda_kernel failed"); }

    return output;
}
