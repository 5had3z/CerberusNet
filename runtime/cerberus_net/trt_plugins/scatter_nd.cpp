#include "scatter_nd.hpp"
#include "trt_utils.hpp"

#include <cassert>
#include <cstring>

namespace
{
    const char* SCATTER_ND_PLUGIN_VERSION{"1"};
    const char* SCATTER_ND_PLUGIN_NAME{"ScatterND"};
} // namespace

nvinfer1::PluginFieldCollection ScatterNDPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> ScatterNDPluginCreator::mPluginAttributes;


ScatterNDPlugin::ScatterNDPlugin(const nvinfer1::PluginFieldCollection& fc)
{
}

ScatterNDPlugin::ScatterNDPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char *>(data), *a = d;

    read(d, m_datatype);

    assert(d == a + length);
}

void ScatterNDPlugin::serialize(void* buffer) const
{
    char* d = static_cast<char*>(buffer), *a = d;

    write(d, static_cast<int>(m_datatype));

    assert(d == a + getSerializationSize());
}

size_t ScatterNDPlugin::getSerializationSize() const
{
    size_t serializationSize = 0;

    serializationSize += sizeof(static_cast<int>(m_datatype));

    return serializationSize;
}

int ScatterNDPlugin::initialize()
{
    return 0;
}

void ScatterNDPlugin::terminate()
{
}

nvinfer1::DimsExprs ScatterNDPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
    int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{
    
}

void ScatterNDPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* ScatterNDPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
nvinfer1::DataType ScatterNDPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // Two inputs and one output
    assert(index == 0 && nbInputs == 3);

    // Data and updates should be the same type
    assert(inputTypes[0] == inputTypes[2]);

    // Second input should be indicies
    assert(inputTypes[1] == nvinfer1::DataType::kINT32);

    return inputTypes[0];
}

bool ScatterNDPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    // Two inputs and one output, only kFLOAT and kHALF Supported
    assert(nbOutputs == 1 && nbInputs == 3);
    // Should be a bog standard tensors
    bool condition = inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;

    if (pos != 1) {
        // Only kFLOAT and kHALF supported as data inputs/updates/outputs
        condition &= (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF);
    }
    else {
        condition &= (inOut[pos].type == nvinfer1::DataType::kINT32);
    }

    // Input and output has same type
    condition &= inOut[pos].type == inOut[nbInputs].type;
    return condition;
}

void ScatterNDPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    assert(in && nbInputs == 2);
    assert(out && nbOutputs == 1);

    // # Check tensor shapes
    // assert indices.shape[-1] <= len(data.shape)
    // assert updates.shape == indices.shape[:-1] + data.shape[indices.shape[-1]:]

    m_datatype = in[0].desc.type;
}

size_t ScatterNDPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
            const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const
{
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void ScatterNDPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void ScatterNDPlugin::detachFromContext()
{
}

const char* ScatterNDPlugin::getPluginType() const
{
    return SCATTER_ND_PLUGIN_NAME;
}

const char* ScatterNDPlugin::getPluginVersion() const
{
    return SCATTER_ND_PLUGIN_VERSION;
}

void ScatterNDPlugin::destroy()
{
    delete this;
}

// Clone the plugin
nvinfer1::IPluginV2DynamicExt* ScatterNDPlugin::clone() const
{
    ScatterNDPlugin *p = new ScatterNDPlugin(*this);
    p->setPluginNamespace(mPluginNamespace);
    return p;
}

ScatterNDPluginCreator::ScatterNDPluginCreator()
{
    mPluginAttributes.clear();

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* ScatterNDPluginCreator::getPluginName() const
{
    return SCATTER_ND_PLUGIN_NAME;
}

const char* ScatterNDPluginCreator::getPluginVersion() const
{
    return SCATTER_ND_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* ScatterNDPluginCreator::getFieldNames()
{
    return &mFC;
}

nvinfer1::IPluginV2DynamicExt* ScatterNDPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc)
{
    ScatterNDPlugin* obj = new ScatterNDPlugin(*fc);
    obj->setPluginNamespace(mNamespace.c_str());
    mPluginName = name;
    mFC = *fc;
    return obj;
}

nvinfer1::IPluginV2DynamicExt* ScatterNDPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    ScatterNDPlugin* obj = new ScatterNDPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
