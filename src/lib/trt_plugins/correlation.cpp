#include "correlation.hpp"
#include "trt_utils.hpp"

#include <cassert>
#include <cstring>

namespace
{
    const char* CORRELATION_PLUGIN_VERSION{"1"};
    const char* CORRELATION_PLUGIN_NAME{"Correlation_TRT"};
} // namespace

nvinfer1::PluginFieldCollection CorrelationPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> CorrelationPluginCreator::mPluginAttributes;

CorrelationPlugin::CorrelationPlugin()
{
}

CorrelationPlugin::CorrelationPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char *>(data), *a = d;
    assert(d == a + length);
}

void CorrelationPlugin::serialize(void* buffer) const
{
    char* d = static_cast<char*>(buffer), *a = d;
    assert(d == a + getSerializationSize());
}

size_t CorrelationPlugin::getSerializationSize() const
{
    return 0;
}

int CorrelationPlugin::initialize()
{
	const int paddedInputHeight = m_input_dims.h() + 2 * m_pad_size;
    const int paddedInputWidth  = m_input_dims.w() + 2 * m_pad_size;
    const int tensor_volume = paddedInputHeight * paddedInputWidth * m_input_dims.c();

    size_t elem_size = 0;
    if (m_datatype == nvinfer1::DataType::kFLOAT) { elem_size = sizeof(float); }
    else if (m_datatype == nvinfer1::DataType::kHALF) { elem_size = sizeof(float) / 2; }

    NV_CUDA_CHECK(cudaMalloc(&m_rInput1, tensor_volume * elem_size));
    NV_CUDA_CHECK(cudaMalloc(&m_rInput2, tensor_volume * elem_size));

    return 0;
}

void CorrelationPlugin::terminate()
{
    NV_CUDA_CHECK(cudaFree(&m_rInput1));
    NV_CUDA_CHECK(cudaFree(&m_rInput2));
}

bool CorrelationPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const
{
    // Two inputs and one output
    assert(nbInputs == 2 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
    bool condition = inOut[pos].format == nvinfer1::TensorFormat::kNCHW;
    condition &= (inOut[pos].type == nvinfer1::DataType::kFLOAT) || (inOut[pos].type == nvinfer1::DataType::kHALF);
    condition &= inOut[pos].type == inOut[nbInputs].type;
    return condition;
}

nvinfer1::Dims CorrelationPlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims)
{
    // Only one output and there are two inputs
    assert(index == 0 && nbInputDims == 2 && inputs[0].nbDims == 3 && inputs[1].nbDims == 3);
    return m_output_dims;
}

void CorrelationPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* CorrelationPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
nvinfer1::DataType CorrelationPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // Only one output and there are two inputs, all of which should have the same datatype
    assert(index == 0 && nbInputs == 2 && inputTypes[0] == m_datatype && inputTypes[1] == m_datatype);
    return m_datatype;
}

// Return true if output tensor is broadcast across a batch.
bool CorrelationPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool CorrelationPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

void CorrelationPlugin::configurePlugin(const nvinfer1::PluginTensorDesc* in, int nbInput, const nvinfer1::PluginTensorDesc* out, int nbOutput)
{
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void CorrelationPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void CorrelationPlugin::detachFromContext()
{
}

const char* CorrelationPlugin::getPluginType() const
{
    return CORRELATION_PLUGIN_NAME;
}

const char* CorrelationPlugin::getPluginVersion() const
{
    return CORRELATION_PLUGIN_VERSION;
}

void CorrelationPlugin::destroy()
{
    delete this;
}

// Clone the plugin
nvinfer1::IPluginV2IOExt* CorrelationPlugin::clone() const
{
    CorrelationPlugin *p = new CorrelationPlugin();
    p->setPluginNamespace(mPluginNamespace);
    return p;
}

CorrelationPluginCreator::CorrelationPluginCreator()
{
    mPluginAttributes.clear();

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* CorrelationPluginCreator::getPluginName() const
{
    return CORRELATION_PLUGIN_NAME;
}

const char* CorrelationPluginCreator::getPluginVersion() const
{
    return CORRELATION_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* CorrelationPluginCreator::getFieldNames()
{
    return &mFC;
}

nvinfer1::IPluginV2IOExt* CorrelationPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc)
{
    assert(!strcmp(name, getPluginName()));
    const nvinfer1::PluginField* fields = fc->fields;

    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
    }

    CorrelationPlugin* obj = new CorrelationPlugin();
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

nvinfer1::IPluginV2IOExt* CorrelationPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    CorrelationPlugin* obj = new CorrelationPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
