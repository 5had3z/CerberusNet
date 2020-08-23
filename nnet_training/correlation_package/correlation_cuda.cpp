#include "correlation_cuda_kernel.cuh"

#include <torch/extension.h>

#include <iostream>

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

std::vector<torch::Tensor> correlation_backward_cuda(
    const torch::Tensor& input1, const torch::Tensor& input2, const torch::Tensor& gradOutput,
    int pad_size, int kernel_size, int max_displacement, int stride1, int stride2, int corr_type_multiply)
{
    auto gradInput1 = torch::zeros_like(input1);
    auto gradInput2 = torch::zeros_like(input2);

    const int success = correlation_backward_cuda_kernel(
        gradOutput, input1, input2, gradInput1, gradInput2, 
        pad_size, kernel_size, max_displacement,
        stride1, stride2, corr_type_multiply);

    if (!success) { AT_ERROR("CUDA correlation_backward_cuda_kernel failed"); }

    return {gradInput1, gradInput2};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &correlation_forward_cuda, "Correlation forward (CUDA)");
    m.def("backward", &correlation_backward_cuda, "Correlation backward (CUDA)");
}
