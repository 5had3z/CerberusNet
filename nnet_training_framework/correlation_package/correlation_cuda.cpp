#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

#include "correlation_cuda_kernel.cuh"

int correlation_forward_cuda(at::Tensor& input1, at::Tensor& input2, at::Tensor& rInput1, at::Tensor& rInput2, at::Tensor& output,
    int pad_size, int kernel_size, int max_displacement, int stride1, int stride2, int corr_type_multiply)
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

    rInput1.resize_({batchSize, paddedInputHeight, paddedInputWidth, nInputChannels});
    rInput2.resize_({batchSize, paddedInputHeight, paddedInputWidth, nInputChannels});
    output.resize_({batchSize, nOutputChannels, outputHeight, outputwidth});

    rInput1.fill_(0);
    rInput2.fill_(0);
    output.fill_(0);

    const int success = correlation_forward_cuda_kernel(
        output, input1, input2, rInput1, rInput2,
        pad_size, kernel_size, max_displacement,
        stride1, stride2, corr_type_multiply,
        at::cuda::getCurrentCUDAStream()
    );

    //check for errors
    if (!success) { AT_ERROR("CUDA call failed"); }

    return 1;
}

int correlation_backward_cuda(at::Tensor& input1, at::Tensor& input2, at::Tensor& rInput1, at::Tensor& rInput2,
    at::Tensor& gradOutput, at::Tensor& gradInput1, at::Tensor& gradInput2,
    int pad_size, int kernel_size, int max_displacement, int stride1, int stride2, int corr_type_multiply)
{
    const int batchSize         = input1.size(0);
    const int nInputChannels    = input1.size(1);
    const int height            = input1.size(2);
    const int width             = input1.size(3);

    const int paddedInputHeight = height + 2 * pad_size;
    const int paddedInputWidth  = width + 2 * pad_size;

    rInput1.resize_({batchSize, paddedInputHeight, paddedInputWidth, nInputChannels});
    rInput2.resize_({batchSize, paddedInputHeight, paddedInputWidth, nInputChannels});
    gradInput1.resize_({batchSize, nInputChannels, height, width});
    gradInput2.resize_({batchSize, nInputChannels, height, width});

    rInput1.fill_(0);
    rInput2.fill_(0);
    gradInput1.fill_(0);
    gradInput2.fill_(0);

    const int success = correlation_backward_cuda_kernel(
        gradOutput, input1, input2, gradInput1, gradInput2, 
        rInput1, rInput2, pad_size, kernel_size, max_displacement,
        stride1, stride2, corr_type_multiply,
        at::cuda::getCurrentCUDAStream()
    );

    if (!success) { AT_ERROR("CUDA call failed"); }

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &correlation_forward_cuda, "Correlation forward (CUDA)");
  m.def("backward", &correlation_backward_cuda, "Correlation backward (CUDA)");
}
