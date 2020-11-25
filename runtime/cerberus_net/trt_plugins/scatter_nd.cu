#include "scatter_nd.hpp"
#include "trt_utils.hpp"
#include "cuda_fp16.h"

#include <limits>
#include <cassert>

template<typename scalar_t, typename intergral_t>
__global__ void ScatterND_Kernel(const scalar_t* __restrict__ source, scalar_t* __restrict__ output,
    intergral_t* __restrict__ indicies, const scalar_t* __restrict__ updates)
{

}

int ScatterNDPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
	const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    // # Compute output
    // output = np.copy(data)
    // for i in np.ndindex(indices.shape[:-1]):
    //     # NOTE: The order of iteration in this loop is not specified.
    //     # In particular, indices should not have duplicate entries: that is, if idx1 != idx2, then indices[idx1] != indices[idx2].
    //     # This ensures that the output value does not depend on the iteration order.
    //     output[indices[i]] = updates[i]
    // return output

    // Temporariy not actually updating input lmoa
    size_t mem_size = inputDesc[0].type == nvinfer1::DataType::kFLOAT ? sizeof(float) : sizeof(__half);
    for (size_t i=0; i<inputDesc[0].dims.nbDims; i++)
    {
        mem_size *= inputDesc[0].dims.d[i];
    }
    NV_CUDA_CHECK(cudaMemcpy(outputs[0], inputs[0], mem_size, cudaMemcpyDeviceToDevice));

    // const size_t nBlocks = 1;
    // const size_t BLOCK_SIZE = 1024;

    // switch (inputDesc[0].type)
    // {
    //     case nvinfer1::DataType::kFLOAT:
    //     {
    //         ScatterND_Kernel<<<nBlocks, BLOCK_SIZE, 0, stream>>>(
    //             reinterpret_cast<const float*>(inputs[0]),
    //             reinterpret_cast<float*>(outputs[0]),
    //             reinterpret_cast<const int32_t*>(inputs[1]),
    //             reinterpret_cast<const float*>(inputs[2]));
    //         break;
    //     }
    //     case nvinfer1::DataType::kHALF:
    //     {
    //         ScatterND_Kernel<<<nBlocks, BLOCK_SIZE, 0, stream>>>(
    //             reinterpret_cast<const __half*>(inputs[0]),
    //             reinterpret_cast<__half*>(outputs[0]),
    //             reinterpret_cast<const int32_t*>(inputs[1]),
    //             reinterpret_cast<const __half*>(inputs[2]));
    //     }
    // }

    const auto error = cudaGetLastError();
    if (error) { std::cout << error << std::endl; }
	NV_CUDA_CHECK(error);
    return error;
}