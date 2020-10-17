#include "grid_sampler.hpp"
#include "trt_utils.hpp"

#include <cassert>
#include <cstring>

namespace
{
    const char* GRID_SAMPLER_PLUGIN_VERSION{"1"};
    const char* GRID_SAMPLER_PLUGIN_NAME{"Grid_Sampler_TRT"};
} // namespace

nvinfer1::PluginFieldCollection GridSamplerPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> GridSamplerPluginCreator::mPluginAttributes;

GridSamplerPlugin::GridSamplerPlugin()
{
}

GridSamplerPlugin::GridSamplerPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char *>(data), *a = d;
    assert(d == a + length);
}

void GridSamplerPlugin::serialize(void* buffer) const
{
    char* d = static_cast<char*>(buffer), *a = d;
    assert(d == a + getSerializationSize());
}

size_t GridSamplerPlugin::getSerializationSize() const
{
    return 0;
}

int GridSamplerPlugin::initialize()
{
    return 0;
}

void GridSamplerPlugin::terminate()
{
}

nvinfer1::Dims GridSamplerPlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims)
{
    //output the result to channel
    return {0, 0, 0};
}

void GridSamplerPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* GridSamplerPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
nvinfer1::DataType GridSamplerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return nvinfer1::DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool GridSamplerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool GridSamplerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

void GridSamplerPlugin::configurePlugin(const nvinfer1::PluginTensorDesc* in, int nbInput, const nvinfer1::PluginTensorDesc* out, int nbOutput)
{
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void GridSamplerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void GridSamplerPlugin::detachFromContext()
{
}

const char* GridSamplerPlugin::getPluginType() const
{
    return GRID_SAMPLER_PLUGIN_NAME;
}

const char* GridSamplerPlugin::getPluginVersion() const
{
    return GRID_SAMPLER_PLUGIN_VERSION;
}

void GridSamplerPlugin::destroy()
{
    delete this;
}

// Clone the plugin
nvinfer1::IPluginV2IOExt* GridSamplerPlugin::clone() const
{
    GridSamplerPlugin *p = new GridSamplerPlugin();
    p->setPluginNamespace(mPluginNamespace);
    return p;
}

GridSamplerPluginCreator::GridSamplerPluginCreator()
{
    mPluginAttributes.clear();

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GridSamplerPluginCreator::getPluginName() const
{
    return GRID_SAMPLER_PLUGIN_NAME;
}

const char* GridSamplerPluginCreator::getPluginVersion() const
{
    return GRID_SAMPLER_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* GridSamplerPluginCreator::getFieldNames()
{
    return &mFC;
}

nvinfer1::IPluginV2IOExt* GridSamplerPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc)
{
    assert(!strcmp(name, getPluginName()));
    const nvinfer1::PluginField* fields = fc->fields;

    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
    }

    GridSamplerPlugin* obj = new GridSamplerPlugin();
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

nvinfer1::IPluginV2IOExt* GridSamplerPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    GridSamplerPlugin* obj = new GridSamplerPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
