#include "scatter_nd.hpp"
#include "trt_utils.hpp"
#include "cuda_fp16.h"

#include <limits>
#include <cassert>

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
    size_t mem_size = sizeof(float);
    for (size_t i=0; i<inputDesc[0].dims.nbDims; i++)
    {
        mem_size *= inputDesc[0].dims.d[i];
    }
    cudaMemcpy(outputs[0], inputs[0], mem_size, cudaMemcpyDeviceToDevice);

    return cudaGetLastError();
}