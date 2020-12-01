#include "grid_sampler.hpp"
#include "trt_utils.hpp"

#include <cassert>
#include <cstring>

namespace
{
    const char* GRID_SAMPLER_PLUGIN_VERSION{"1"};
    const char* GRID_SAMPLER_PLUGIN_NAME{"grid_sampler"};
} // namespace

nvinfer1::PluginFieldCollection GridSamplerPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> GridSamplerPluginCreator::mPluginAttributes;

GridSamplerPlugin::GridSamplerPlugin(const nvinfer1::PluginFieldCollection& fc)
{
    for (int i = 0; i < fc.nbFields; ++i)
    {
        const char* attrName = fc.fields[i].name;
        if (!strcmp(attrName, "align_corners"))
        {
            assert(fc.fields[i].type == nvinfer1::PluginFieldType::kINT32);
            m_align_corners = *(static_cast<const bool*>(fc.fields[i].data));
        }
        else if (!strcmp(attrName, "interpolation_mode"))
        {
            assert(fc.fields[i].type == nvinfer1::PluginFieldType::kINT32);
            m_interpolation_mode = *(static_cast<const GridSampler::Interpolation*>(fc.fields[i].data));
        }
        else if (!strcmp(attrName, "padding_mode"))
        {
            assert(fc.fields[i].type == nvinfer1::PluginFieldType::kINT32);
            m_padding_mode = *(static_cast<const GridSampler::Padding*>(fc.fields[i].data));
        }
    }

    if (!fc.nbFields) {
        std::cerr << "Grid_sampler Plugin: No fields detected, using default parameters";
        m_align_corners = false;
        m_interpolation_mode = Interpolation::Bilinear;
        m_padding_mode = Padding::Border;
    }
}

GridSamplerPlugin::GridSamplerPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char *>(data), *a = d;

    read(d, m_align_corners);
    read(d, m_interpolation_mode);
    read(d, m_padding_mode);

    assert(d == a + length);
}

void GridSamplerPlugin::serialize(void* buffer) const
{
    char* d = static_cast<char*>(buffer), *a = d;

    write(d, m_align_corners);
    write(d, static_cast<int>(m_interpolation_mode));
    write(d, static_cast<int>(m_padding_mode));

    assert(d == a + getSerializationSize());
}

size_t GridSamplerPlugin::getSerializationSize() const
{
    size_t serializationSize = 0;

    serializationSize += sizeof(m_align_corners);
    serializationSize += sizeof(static_cast<int>(m_interpolation_mode));
    serializationSize += sizeof(static_cast<int>(m_padding_mode));

    return serializationSize;
}

int GridSamplerPlugin::initialize()
{
    return 0;
}

void GridSamplerPlugin::terminate()
{
}

size_t GridSamplerPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
    int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const
{
    // No additional workspace required, ops done inplace.
    return 0;
}

nvinfer1::DimsExprs GridSamplerPlugin::getOutputDimensions(int outputIndex,
    const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{
    // Only one input and one output which should match dimensions
    assert(outputIndex == 0 && nbInputs == 2);
    return inputs[0];
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
    // Two inputs and one output, only kFLOAT and kHALF Supported
    assert(index == 0 && nbInputs == 2);
    for (int i = 0; i < nbInputs; ++i)
    {
        assert(inputTypes[i] == nvinfer1::DataType::kFLOAT || inputTypes[i] == nvinfer1::DataType::kHALF);
    }
    return inputTypes[index];
}

bool GridSamplerPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    // Two inputs and one output, only kFLOAT and kHALF Supported
    assert(nbOutputs == 1 && nbInputs == 2);
    bool condition = 1;
    // Should be a bog standard tensor however format doesn't really matter I guess?
    condition &= inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    // Only kFLOAT and kHALF supported
    condition &= (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF);
    // Input and output has same type except if the end is dynamic
    condition &= (inOut[pos].type == inOut[nbInputs].type || (int32_t)inOut[nbInputs].type == -1);
    condition &= (inOut[0].dims.d[2] == inOut[1].dims.d[1]);
    condition &= (inOut[0].dims.d[3] == inOut[1].dims.d[2]);
    condition &= (inOut[1].dims.d[3] == 2);
    if (pos == 2) {
        condition &= inOut[0].type == inOut[2].type && inOut[0].type == inOut[1].type;
    }
    return condition;
}

void GridSamplerPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    assert(in && nbInputs == 2);
    assert(out && nbOutputs == 1);
    assert(in[0].desc.type == in[1].desc.type && in[0].desc.type == out[0].desc.type);

    assert(in[0].desc.dims.nbDims == in[1].desc.dims.nbDims);
    
    // Input1: NCHW, Input2: NHW2
    assert(in[0].desc.dims.d[0] == in[1].desc.dims.d[0]);
    assert(in[0].desc.dims.d[2] == in[1].desc.dims.d[1]);
    assert(in[0].desc.dims.d[3] == in[1].desc.dims.d[2]);
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
nvinfer1::IPluginV2DynamicExt* GridSamplerPlugin::clone() const
{
    GridSamplerPlugin *p = new GridSamplerPlugin(*this);
    p->setPluginNamespace(mPluginNamespace);
    return p;
}

GridSamplerPluginCreator::GridSamplerPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(nvinfer1::PluginField("align_corners", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("interpolation_mode", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("padding_mode", nullptr, nvinfer1::PluginFieldType::kINT32, 1));

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

nvinfer1::IPluginV2DynamicExt* GridSamplerPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc)
{
    GridSamplerPlugin* obj = new GridSamplerPlugin(*fc);
    obj->setPluginNamespace(mNamespace.c_str());
    mPluginName = name;
    return obj;
}

nvinfer1::IPluginV2DynamicExt* GridSamplerPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    GridSamplerPlugin* obj = new GridSamplerPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
