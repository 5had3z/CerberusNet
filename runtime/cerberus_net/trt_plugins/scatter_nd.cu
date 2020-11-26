#include "scatter_nd.hpp"
#include "trt_utils.hpp"
#include "cuda_fp16.h"

#include <array>
#include <algorithm>
#include <limits>
#include <cassert>

int ScatterNDPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
	const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    // Source code from ONNX repo implementation
    // # Compute output
    // output = np.copy(data)
    // for i in np.ndindex(indices.shape[:-1]):
    //     # NOTE: The order of iteration in this loop is not specified.
    //     # In particular, indices should not have duplicate entries: that is, if idx1 != idx2, then indices[idx1] != indices[idx2].
    //     # This ensures that the output value does not depend on the iteration order.
    //     output[indices[i]] = updates[i]
    // return output

    // Copy the entire input to the output before update
    size_t mem_size = inputDesc[0].type == nvinfer1::DataType::kFLOAT ? sizeof(float) : sizeof(__half);
    for (size_t i=0; i<inputDesc[0].dims.nbDims; i++)
    {
        mem_size *= inputDesc[0].dims.d[i];
    }
    NV_CUDA_CHECK(cudaMemcpy(outputs[0], inputs[0], mem_size, cudaMemcpyDeviceToDevice));

    // Checking what is actually being given as update indicies, its seems like the channel stays constant
    // and the elements are being iterated over, therefore we should just be able to copy the entire channel.
    // std::cout << "ScatterND\n";
    // for (size_t i=0; i<130; i++)
    // {
    //     std::array<int32_t, 4> data;
    //     const size_t dim_stride = 4 * sizeof(int32_t);
    //     NV_CUDA_CHECK(cudaMemcpy(data.data(), inputs[1] + i * dim_stride, dim_stride, cudaMemcpyDeviceToHost));
    
    //     std::for_each(data.begin(), data.end(), [](const auto& elem){ std::cout << elem << ", ";} );
    //     std::cout << "\n";
    // }
    // std::cout << std::endl;

    std::array<int32_t, 4> indicies_element;
    const size_t dim_stride = 4 * sizeof(int32_t);
    NV_CUDA_CHECK(cudaMemcpy(indicies_element.data(), inputs[1], dim_stride, cudaMemcpyDeviceToHost));

    const size_t elem_size = inputDesc[0].type == nvinfer1::DataType::kFLOAT ? sizeof(float) : sizeof(__half);
    const size_t channel_stride = inputDesc[0].dims.d[2] * inputDesc[0].dims.d[3] * elem_size;
    
    // indicies_element[1] contains the channel indx that is being updated.
    NV_CUDA_CHECK(cudaMemcpy(
        outputs[0] + indicies_element[1] * elem_size * channel_stride, inputs[2],
        elem_size * channel_stride, cudaMemcpyDeviceToDevice));

    return cudaGetLastError();
}
