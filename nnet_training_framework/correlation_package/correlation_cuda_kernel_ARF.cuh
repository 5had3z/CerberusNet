#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

int correlation_forward_cuda_kernel(torch::Tensor& output, torch::Tensor& input1,
    torch::Tensor& input2, torch::Tensor& rInput1, torch::Tensor& rInput2,
    int pad_size, int kernel_size, int max_displacement, int stride1,
    int stride2, int corr_type_multiply, cudaStream_t stream);

int correlation_backward_cuda_kernel(torch::Tensor& gradOutput, torch::Tensor& input1,
    torch::Tensor& input2, torch::Tensor& gradInput1, torch::Tensor& gradInput2,
    torch::Tensor& rInput1, torch::Tensor& rInput2,
    int pad_size, int kernel_size, int max_displacement, int stride1,
    int stride2, int corr_type_multiply, cudaStream_t stream);
