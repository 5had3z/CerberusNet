#pragma once

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <cuda_runtime.h>

int correlation_forward_cuda_kernel(at::Tensor& output, at::Tensor& input1,
    at::Tensor& input2, at::Tensor& rInput1, at::Tensor& rInput2,
    int pad_size, int kernel_size, int max_displacement, int stride1,
    int stride2, int corr_type_multiply, cudaStream_t stream);

int correlation_backward_cuda_kernel(at::Tensor& gradOutput, at::Tensor& input1,
    at::Tensor& input2, at::Tensor& gradInput1, at::Tensor& gradInput2,
    at::Tensor& rInput1, at::Tensor& rInput2,
    int pad_size, int kernel_size, int max_displacement, int stride1,
    int stride2, int corr_type_multiply, cudaStream_t stream);
