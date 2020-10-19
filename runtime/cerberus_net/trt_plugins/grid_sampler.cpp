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
        if (!strcmp(attrName, "interpolation_mode"))
        {
            assert(fc.fields[i].type == nvinfer1::PluginFieldType::kINT32);
            m_interpolation_mode = *(static_cast<const GridSampler::Interpolation*>(fc.fields[i].data));
        }
        if (!strcmp(attrName, "padding_mode"))
        {
            assert(fc.fields[i].type == nvinfer1::PluginFieldType::kINT32);
            m_padding_mode = *(static_cast<const GridSampler::Padding*>(fc.fields[i].data));
        }
    }
}

GridSamplerPlugin::GridSamplerPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char *>(data), *a = d;

    read(d, m_input_dims.nbDims);
    assert(m_input_dims.nbDims <= m_input_dims.MAX_DIMS);
    for (int i = 0; i < m_input_dims.nbDims; ++i)
    {
        read(d, m_input_dims.d[i]);
    }

    read(d, m_output_dims.nbDims);
    assert(m_output_dims.nbDims <= m_output_dims.MAX_DIMS);
    for (int i = 0; i < m_output_dims.nbDims; ++i)
    {
        read(d, m_output_dims.d[i]);
    }

    read(d, m_align_corners);
    read(d, m_interpolation_mode);
    read(d, m_padding_mode);
    read(d, m_datatype);

    assert(d == a + length);
}

void GridSamplerPlugin::serialize(void* buffer) const
{
    char* d = static_cast<char*>(buffer), *a = d;

    write(d, m_input_dims.nbDims);
    assert(m_input_dims.nbDims <= m_input_dims.MAX_DIMS);
    for (int i = 0; i < m_input_dims.nbDims; ++i)
    {
        write(d, m_input_dims.d[i]);
    }

    write(d, m_output_dims.nbDims);
    assert(m_output_dims.nbDims <= m_output_dims.MAX_DIMS);
    for (int i = 0; i < m_output_dims.nbDims; ++i)
    {
        write(d, m_output_dims.d[i]);
    }

    write(d, static_cast<int>(m_align_corners));
    write(d, static_cast<int>(m_interpolation_mode));
    write(d, static_cast<int>(m_padding_mode));
    write(d, static_cast<int>(m_datatype));

    assert(d == a + getSerializationSize());
}

size_t GridSamplerPlugin::getSerializationSize() const
{
    size_t serializationSize = 0;

    serializationSize += sizeof(m_input_dims.nbDims);
    serializationSize += sizeof(m_input_dims.d[0]) * m_input_dims.nbDims;
    serializationSize += sizeof(m_output_dims.nbDims);
    serializationSize += sizeof(m_output_dims.d[0]) * m_output_dims.nbDims;
    serializationSize += sizeof(static_cast<int>(m_align_corners));
    serializationSize += sizeof(static_cast<int>(m_interpolation_mode));
    serializationSize += sizeof(static_cast<int>(m_padding_mode));
    serializationSize += sizeof(static_cast<int>(m_datatype));

    return serializationSize;
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
    // Only one input and one output
    assert(index == 0 && nbInputDims == 2);
    return m_output_dims;
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
        assert(inputTypes[i] == nvinfer1::DataType::kFLOAT); // || inputTypes[i] == nvinfer1::DataType::kHALF);
    }
    return inputTypes[index];
}

bool GridSamplerPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const
{
    // Two inputs and one output, only kFLOAT and kHALF Supported
    assert(nbOutputs == 1 && nbInputs == 2);
    // Should be a bog standard tensor
    bool condition = inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    // Only kFLOAT and kHALF supported
    condition &= (inOut[pos].type == nvinfer1::DataType::kFLOAT); // || (inOut[pos].type == nvinfer1::DataType::kHALF);
    // Input and output has same type
    condition &= inOut[pos].type == inOut[nbInputs].type;
    return condition;
}

// Return true if output tensor is broadcast across a batch.
bool GridSamplerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return true;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool GridSamplerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return true;
}

void GridSamplerPlugin::configurePlugin(const nvinfer1::PluginTensorDesc* in, int nbInput, const nvinfer1::PluginTensorDesc* out, int nbOutput)
{
    assert(in && nbInput == 2);
    assert(out && nbOutput == 1);
    assert(in[0].type == in[1].type && in[0].type == out[0].type);

    assert(in[0].dims.d == in[1].dims.d);

    m_datatype = in[0].type;
    m_input_dims = in[0].dims;
    m_output_dims = out[0].dims;
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
    GridSamplerPlugin *p = new GridSamplerPlugin(*this);
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
    GridSamplerPlugin* obj = new GridSamplerPlugin(*fc);
    obj->setPluginNamespace(mNamespace.c_str());
    mFC = *fc;
    return obj;
}

nvinfer1::IPluginV2IOExt* GridSamplerPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    GridSamplerPlugin* obj = new GridSamplerPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
