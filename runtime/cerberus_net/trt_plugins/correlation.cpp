#include "correlation.hpp"
#include "trt_utils.hpp"

#include <cmath>
#include <cassert>
#include <cstring>

namespace
{
    const char* CORRELATION_PLUGIN_VERSION{"1"};
    const char* CORRELATION_PLUGIN_NAME{"correlation"};
} // namespace

nvinfer1::PluginFieldCollection CorrelationPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> CorrelationPluginCreator::mPluginAttributes;

CorrelationPlugin::CorrelationPlugin(const nvinfer1::PluginFieldCollection& fc)
{
    for (int i = 0; i < fc.nbFields; ++i)
    {
        const char* attrName = fc.fields[i].name;
        if (!strcmp(attrName, "pad_size"))
        {
            assert(fc.fields[i].type == nvinfer1::PluginFieldType::kINT32);
            m_pad_size = *(static_cast<const int*>(fc.fields[i].data));
        }
        else if (!strcmp(attrName, "kernel_size"))
        {
            assert(fc.fields[i].type == nvinfer1::PluginFieldType::kINT32);
            m_kernel_size = *(static_cast<const int*>(fc.fields[i].data));
        }
        else if (!strcmp(attrName, "max_displacement"))
        {
            assert(fc.fields[i].type == nvinfer1::PluginFieldType::kINT32);
            m_max_displacement = *(static_cast<const int*>(fc.fields[i].data));
        }
        else if (!strcmp(attrName, "stride1"))
        {
            assert(fc.fields[i].type == nvinfer1::PluginFieldType::kINT32);
            m_stride1 = *(static_cast<const int*>(fc.fields[i].data));
        }
        else if (!strcmp(attrName, "stride2"))
        {
            assert(fc.fields[i].type == nvinfer1::PluginFieldType::kINT32);
            m_stride2 = *(static_cast<const int*>(fc.fields[i].data));
        }
        else if (!strcmp(attrName, "corr_multiply"))
        {
            assert(fc.fields[i].type == nvinfer1::PluginFieldType::kINT32);
            m_corr_multiply = *(static_cast<const int*>(fc.fields[i].data));
        }
    }

    if (!fc.nbFields){
        std::cerr << "Correlation Plugin: No fields detected, using default parameters";
        m_pad_size = 4;
        m_kernel_size = 1;
        m_max_displacement = 4;
        m_stride1 = 1;
        m_stride2 = 1;
        m_corr_multiply = 1;
    }
}

CorrelationPlugin::CorrelationPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char *>(data), *a = d;

    read(d, m_pad_size);
    read(d, m_kernel_size);
    read(d, m_max_displacement);
    read(d, m_stride1);
    read(d, m_stride2);
    read(d, m_corr_multiply);

    assert(d == a + length);
}

void CorrelationPlugin::serialize(void* buffer) const
{
    char* d = static_cast<char*>(buffer), *a = d;

    write(d, m_pad_size);
    write(d, m_kernel_size);
    write(d, m_max_displacement);
    write(d, m_stride1);
    write(d, m_stride2);
    write(d, m_corr_multiply);

    assert(d == a + getSerializationSize());
}

size_t CorrelationPlugin::getSerializationSize() const
{
    size_t serializationSize = 0;

    serializationSize += sizeof(m_pad_size);
    serializationSize += sizeof(m_kernel_size);
    serializationSize += sizeof(m_max_displacement);
    serializationSize += sizeof(m_stride1);
    serializationSize += sizeof(m_stride2);
    serializationSize += sizeof(m_corr_multiply);

    return serializationSize;
}

int CorrelationPlugin::initialize()
{
    NV_CUDA_CHECK(cudaStreamCreate(&stream_a));
	NV_CUDA_CHECK(cudaStreamCreate(&stream_b));
    return 0;
}

size_t CorrelationPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
    int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const
{
    assert(inputs[0].dims.nbDims == inputs[1].dims.nbDims);
    for (int i=0; i<inputs[0].dims.nbDims; i++)
    {
        assert(inputs[0].dims.d[i] == inputs[1].dims.d[i]);
    }

    // Input descriptors [batch, channels, height, width]
    const size_t paddedInputHeight = inputs[0].dims.d[2] + 2 * m_pad_size;
    const size_t paddedInputWidth  = inputs[0].dims.d[3] + 2 * m_pad_size;
    size_t tensor_volume = inputs[0].dims.d[0] * inputs[0].dims.d[1] * paddedInputHeight * paddedInputWidth;

    switch (inputs[0].type)
    {
        case nvinfer1::DataType::kFLOAT: { tensor_volume *= sizeof(float); break; }
        case nvinfer1::DataType::kHALF: { tensor_volume *= sizeof(float) / 2; break; }
        default: { std::logic_error( "Invalid type used in correlation only float and half are supported" ); }
    }

    return 2 * tensor_volume;
}

void CorrelationPlugin::terminate()
{
    NV_CUDA_CHECK(cudaStreamDestroy(stream_a));
	NV_CUDA_CHECK(cudaStreamDestroy(stream_b));
}

bool CorrelationPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    // Two inputs and one output
    assert(nbInputs == 2 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
    // Should be bog standard tensors
    bool condition = inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    // Only kFLOAT and kHALF supported
    condition &= (inOut[pos].type == nvinfer1::DataType::kFLOAT) || (inOut[pos].type == nvinfer1::DataType::kHALF);
    // Input and output has same type unless output is now dynamic
    condition &= (inOut[pos].type == inOut[nbInputs].type || (int32_t)inOut[nbInputs].type == -1);

    // Both inputs have same dimensions
    for (int i=0; i<inOut[0].dims.nbDims; i++)
    {
        condition &= inOut[0].dims.d[i] == inOut[1].dims.d[i];
    }
    return condition;
}

nvinfer1::DimsExprs CorrelationPlugin::getOutputDimensions(int outputIndex,
    const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{
    // Only one output and there are eight inputs (two dimensions and six args)
    assert(nbInputs == 2 && outputIndex == 0);
    
    // Should be NCHW
    assert(inputs[0].nbDims == 4 && inputs[1].nbDims == 4);

    const int32_t kernel_radius = (m_kernel_size - 1) / 2;
    const int32_t border_radius = kernel_radius + m_max_displacement;

    nvinfer1::DimsExprs outdims;
    outdims.nbDims = 4;
    outdims.d[0] =  inputs[0].d[0];
    outdims.d[1] =  exprBuilder.constant(((m_max_displacement / m_stride2)*2+1) * ((m_max_displacement / m_stride2)*2+1));
    outdims.d[2] =  exprBuilder.operation(nvinfer1::DimensionOperation::kCEIL_DIV,
                        *exprBuilder.operation(nvinfer1::DimensionOperation::kSUB,
                            *exprBuilder.operation(nvinfer1::DimensionOperation::kSUM, *inputs[0].d[2], *exprBuilder.constant(2 * m_pad_size)),
                        *exprBuilder.constant(2 * border_radius)),
                    *exprBuilder.constant(m_stride1));
    outdims.d[3] = exprBuilder.operation(nvinfer1::DimensionOperation::kCEIL_DIV,
                        *exprBuilder.operation(nvinfer1::DimensionOperation::kSUB,
                            *exprBuilder.operation(nvinfer1::DimensionOperation::kSUM, *inputs[0].d[3], *exprBuilder.constant(2 * m_pad_size)),
                        *exprBuilder.constant(2 * border_radius)),
                    *exprBuilder.constant(m_stride1));
    return outdims;
}

void CorrelationPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    // Only one output and there are eight inputs (two dimensions and six args)
    assert(nbInputs == 2 && nbOutputs == 1);
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
    assert(index == 0 && inputTypes[0] == inputTypes[1]);
    return inputTypes[0];
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
nvinfer1::IPluginV2DynamicExt* CorrelationPlugin::clone() const
{
    CorrelationPlugin *p = new CorrelationPlugin(*this);
    p->setPluginNamespace(mPluginNamespace);
    return p;
}

CorrelationPluginCreator::CorrelationPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(nvinfer1::PluginField("pad_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("kernel_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("max_displacement", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("stride1", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("stride2", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("corr_multiply", nullptr, nvinfer1::PluginFieldType::kINT32, 1));

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

nvinfer1::IPluginV2DynamicExt* CorrelationPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc)
{
    CorrelationPlugin* obj = new CorrelationPlugin(*fc);
    obj->setPluginNamespace(mNamespace.c_str());
    mPluginName = name;
    return obj;
}

nvinfer1::IPluginV2DynamicExt* CorrelationPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    CorrelationPlugin* obj = new CorrelationPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
