#pragma once
#include <torch/extension.h>
#include <THC/THC.h>

void blob_rearrange_ongpu(const torch::Tensor& in, torch::Tensor& out, 
    int num, int channels, int width, int height,
    int widthheight, int padding, int pwidthheight, cudaStream_t stream);

void CorrelateData_ongpu(const torch::Tensor& rbot1, const torch::Tensor& rbot2,
    torch::Tensor& output, int batchSize, int nOutputCols, int nOutputRows, int nOutputPlane,
    int max_displacement, int neighborhood_grid_radius_, int neighborhood_grid_width_,
    int kernel_radius_, int kernel_size, int stride1, int stride2, int paddedbottomwidth,
    int paddedbottomheight, int nInputPlane, int corr_type_multiply, cudaStream_t stream);

void CorrelateDataBackward_ongpu(const torch::Tensor& rbot1, const torch::Tensor& rbot2,
    const torch::Tensor& gradOutput, torch::Tensor& gradInput1, torch::Tensor& gradInput2,
    int batchSize, int nOutputCols, int nOutputRows, int nOutputPlane,
    int max_displacement, int neighborhood_grid_radius_, int neighborhood_grid_width_,
    int kernel_radius_, int stride1, int stride2, int nInputCols, int nInputRows,
    int paddedbottomwidth, int paddedbottomheight, int nInputPlane, int pad_size,
    int corr_type_multiply, cudaStream_t stream);
