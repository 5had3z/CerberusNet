#include "correlation_cuda_kernel_PWC.cuh"
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

// == Forward
int corr_cuda_forward(torch::Tensor &input1, torch::Tensor &input2,
    torch::Tensor &rbot1, torch::Tensor &rbot2, torch::Tensor &output,
    int pad_size, int kernel_size, int max_displacement,
    int stride1, int stride2, int corr_type_multiply)
{
    std::cout << "going forward" << std::endl;

    int batchSize = input1.size(0);
    long nInputPlane = input1.size(1);
    long nInputRows = input1.size(2);
    long nInputCols = input1.size(3);
    long inputWidthHeight = nInputRows * nInputCols;

    long kernel_radius_ = (kernel_size - 1) / 2;
    long border_size_ = max_displacement + kernel_radius_; // size of unreachable border region (on each side)

    long paddedbottomheight = nInputRows + 2 * pad_size;
    long paddedbottomwidth = nInputCols + 2 * pad_size;

    long nOutputCols = ceil((float)(paddedbottomwidth - border_size_ * 2) / (float)stride1);
    long nOutputRows = ceil((float)(paddedbottomheight - border_size_ * 2) / (float)stride1);

    // Given a center position in image 1, how many displaced positions in -x / +x
    // direction do we consider in image2 (neighborhood_grid_width)
    long neighborhood_grid_radius_ = max_displacement / stride2;
    long neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;

    // Number of output channels amounts to displacement combinations in X and Y direction
    int nOutputPlane = neighborhood_grid_width_ * neighborhood_grid_width_;

    // Outputs
    output.resize_({batchSize, nOutputPlane, nOutputRows, nOutputCols});
    output.fill_({0});

    rbot1.resize_({batchSize, nInputPlane, paddedbottomheight, paddedbottomwidth});
    rbot1.fill_({0});
    rbot2.resize_({batchSize, nInputPlane, paddedbottomheight, paddedbottomwidth});
    rbot2.fill_({0});

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int pwidthheight = paddedbottomwidth * paddedbottomheight;

    blob_rearrange_ongpu(input1.data<float>(), rbot1.data<float>(),
        batchSize, nInputPlane, nInputCols, nInputRows,
        inputWidthHeight, pad_size, pwidthheight, stream);

    blob_rearrange_ongpu(input2.data<float>(), rbot2.data<float>(),
        batchSize, nInputPlane, nInputCols, nInputRows,
        inputWidthHeight, pad_size, pwidthheight, stream);

    CorrelateData_ongpu(rbot1.data<float>(), rbot2.data<float>(), output.data<float>(),
        batchSize, nOutputCols, nOutputRows, nOutputPlane, max_displacement,
        neighborhood_grid_radius_, neighborhood_grid_width_, kernel_radius_,
        kernel_size, stride1, stride2, paddedbottomwidth, paddedbottomheight,
        nInputPlane, corr_type_multiply, stream);

    return 1;
}

int corr_cuda_backward(torch::Tensor &input1, torch::Tensor &input2,
    torch::Tensor &rbot1, torch::Tensor &rbot2, torch::Tensor &gradOutput,
    torch::Tensor &gradInput1, torch::Tensor &gradInput2,
    int pad_size, int kernel_size, int max_displacement,
    int stride1, int stride2, int corr_type_multiply)
{
    std::cout << "going backward" << std::endl;
    
    long nInputCols = input1.size(3);
    long nInputRows = input1.size(2);
    long nInputPlane = input1.size(1);
    long batchSize = input1.size(0);

    long inputWidthHeight = nInputRows * nInputCols;

    long kernel_radius_ = (kernel_size - 1) / 2;
    long border_size_ = max_displacement + kernel_radius_; // size of unreachable border region (on each side)

    long paddedbottomheight = nInputRows + 2 * pad_size;
    long paddedbottomwidth = nInputCols + 2 * pad_size;

    long nOutputCols = ceil((float)(paddedbottomwidth - border_size_ * 2) / (float)stride1);
    long nOutputRows = ceil((float)(paddedbottomheight - border_size_ * 2) / (float)stride1);

    // Given a center position in image 1, how many displaced positions in -x / +x
    // direction do we consider in image2 (neighborhood_grid_width)
    long neighborhood_grid_radius_ = max_displacement / stride2;
    long neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;

    // Number of output channels amounts to displacement combinations in X and Y direction
    const int nOutputPlane = neighborhood_grid_width_ * neighborhood_grid_width_;

    // Resize rearrange
    rbot1.resize_({batchSize, nInputPlane, paddedbottomheight, paddedbottomwidth});
    rbot1.fill_({0});
    rbot2.resize_({batchSize, nInputPlane, paddedbottomheight, paddedbottomwidth});
    rbot2.fill_({0});

    // Resize grad of inputs
    gradInput1.resize_(input1.sizes());
    gradInput1.fill_({0});
    gradInput2.resize_(input2.sizes());
    gradInput2.fill_({0});

    const int pwidthheight = paddedbottomwidth * paddedbottomheight;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    blob_rearrange_ongpu(input1.data<float>(), rbot1.data<float>(),
        batchSize, nInputPlane, nInputCols, nInputRows,
        inputWidthHeight, pad_size, pwidthheight, stream);

    blob_rearrange_ongpu(input2.data<float>(), rbot2.data<float>(),
        batchSize, nInputPlane, nInputCols, nInputRows,
        inputWidthHeight, pad_size, pwidthheight, stream);

    // CorrelationLayerBackward

    CorrelateDataBackward_ongpu(rbot1.data<float>(), rbot2.data<float>(),
        gradOutput.data<float>(), gradInput1.data<float>(), gradInput2.data<float>(),
        batchSize, nOutputCols, nOutputRows, nOutputPlane, max_displacement,
        neighborhood_grid_radius_, neighborhood_grid_width_, kernel_radius_,
        stride1, stride2, nInputCols, nInputRows, paddedbottomwidth, paddedbottomheight,
        nInputPlane, pad_size, corr_type_multiply, stream);

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &corr_cuda_forward, "Correlation forward PWC (CUDA)");
  m.def("backward", &corr_cuda_backward, "Correlation backward PWC (CUDA)");
}
