#pragma once

#include <torch/extension.h>

int correlation_forward_cuda_kernel(torch::Tensor& output,
    const torch::Tensor& input1, const torch::Tensor& input2,
    const int pad_size, const int kernel_size, const int max_displacement,
    const int stride1, const int stride2, const int corr_type_multiply);

int correlation_backward_cuda_kernel(const torch::Tensor& gradOutput,
    const torch::Tensor& input1, const torch::Tensor& input2,
    torch::Tensor& gradInput1, torch::Tensor& gradInput2,
    const int pad_size, const int kernel_size, const int max_displacement,
    const int stride1, const int stride2, const int corr_type_multiply);
