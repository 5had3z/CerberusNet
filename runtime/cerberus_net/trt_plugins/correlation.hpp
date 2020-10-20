#pragma once

#include <cuda_runtime.h>

#include <vector>
#include <string>

#include <NvInfer.h>

class CorrelationPlugin: public nvinfer1::IPluginV2DynamicExt
{
    public:
        CorrelationPlugin(const nvinfer1::PluginFieldCollection& fc);

        CorrelationPlugin(const void* data, size_t length);

        ~CorrelationPlugin() override = default;

        int getNbOutputs() const override { return 1; }

        nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
            int nbInputs, nvinfer1::IExprBuilder& exprBuilder) override;

        int initialize() override;

        void terminate() override;

        size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
            const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;

        int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
            const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

        void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
            const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;

        size_t getSerializationSize() const override;

        void serialize(void* buffer) const override;

        bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;

        const char* getPluginType() const override;

        const char* getPluginVersion() const override;

        void destroy() override;

        IPluginV2DynamicExt* clone() const override;

        void setPluginNamespace(const char* pluginNamespace) override;

        const char* getPluginNamespace() const override;

        nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

        void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator) override;

        void detachFromContext() override;

    private:
        nvinfer1::DataType m_datatype;

        int m_pad_size;
        int m_kernel_size;
        int m_max_displacement;
        int m_stride1;
        int m_stride2;
        int m_corr_multiply;

        nvinfer1::Dims m_input_dims;
        nvinfer1::Dims m_output_dims;

        void* m_rInput1;
        void* m_rInput2;

        const char* mPluginNamespace;
};

class CorrelationPluginCreator : public nvinfer1::IPluginCreator
{
    public:
        CorrelationPluginCreator();

        ~CorrelationPluginCreator() override = default;

        const char* getPluginName() const override;

        const char* getPluginVersion() const override;

        const nvinfer1::PluginFieldCollection* getFieldNames() override;

        nvinfer1::IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override;

        nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

        void setPluginNamespace(const char* libNamespace) override
        {
            mNamespace = libNamespace;
        }

        const char* getPluginNamespace() const override
        {
            return mNamespace.c_str();
        }

    private:
        static nvinfer1::PluginFieldCollection mFC;
        static std::vector<nvinfer1::PluginField> mPluginAttributes;
        std::string mNamespace;
        std::string mPluginName;
};

REGISTER_TENSORRT_PLUGIN(CorrelationPluginCreator);