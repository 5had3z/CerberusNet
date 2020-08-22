#include "correlation_cuda_kernel_ARF.cuh"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <iostream>

int correlation_forward_cuda(torch::Tensor& input1, torch::Tensor& input2, torch::Tensor& rInput1, torch::Tensor& rInput2, torch::Tensor& output,
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

int correlation_backward_cuda(torch::Tensor& input1, torch::Tensor& input2, torch::Tensor& rInput1, torch::Tensor& rInput2,
    torch::Tensor& gradOutput, torch::Tensor& gradInput1, torch::Tensor& gradInput2,
    int pad_size, int kernel_size, int max_displacement, int stride1, int stride2, int corr_type_multiply)
{
    const int batchSize         = input1.size(0);
    const int nInputChannels    = input1.size(1);
    const int inputHeight       = input1.size(2);
    const int inputWidth        = input1.size(3);

    std::cout << "Input batch: " << batchSize << " ch: " << nInputChannels << 
        " h: " << inputHeight << " w: " << inputWidth << std::endl;

    std::cout << "Input strides batch: " << input1.stride(0) << " ch: " << input1.stride(1) << 
        " h: " << input1.stride(2) << " w: " << input1.stride(3) << std::endl;

    const int paddedInputHeight = inputHeight + 2 * pad_size;
    const int paddedInputWidth  = inputWidth + 2 * pad_size;

    rInput1.resize_({batchSize, paddedInputHeight, paddedInputWidth, nInputChannels}, c10::MemoryFormat::Contiguous);
    rInput2.resize_({batchSize, paddedInputHeight, paddedInputWidth, nInputChannels}, c10::MemoryFormat::Contiguous);
    gradInput1.resize_({batchSize, nInputChannels, inputHeight, inputWidth}, c10::MemoryFormat::Contiguous);
    gradInput2.resize_({batchSize, nInputChannels, inputHeight, inputWidth}, c10::MemoryFormat::Contiguous);

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
  m.def("forward", &correlation_forward_cuda, "Correlation forward ARF (CUDA)");
  m.def("backward", &correlation_backward_cuda, "Correlation backward ARF (CUDA)");
}
