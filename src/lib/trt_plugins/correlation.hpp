#pragma once

#include <cuda_runtime.h>

#include <vector>
#include <string>

#include <NvInfer.h>

class CorrelationPlugin: public nvinfer1::IPluginV2IOExt
{
    public:
        CorrelationPlugin();

        CorrelationPlugin(const void* data, size_t length);

        ~CorrelationPlugin() override = default;

        int getNbOutputs() const override
        {
            return 1;
        }

        nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override;

        int initialize() override;

        void terminate() override;

        virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0;}

        virtual int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

        virtual size_t getSerializationSize() const override;

        virtual void serialize(void* buffer) const override;

        bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override;

        const char* getPluginType() const override;

        const char* getPluginVersion() const override;

        void destroy() override;

        IPluginV2IOExt* clone() const override;

        void setPluginNamespace(const char* pluginNamespace) override;

        const char* getPluginNamespace() const override;

        nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

        bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

        bool canBroadcastInputAcrossBatch(int inputIndex) const override;

        void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator) override;

        void configurePlugin(const nvinfer1::PluginTensorDesc* in, int nbInput, const nvinfer1::PluginTensorDesc* out, int nbOutput) override TRTNOEXCEPT;

        void detachFromContext() override;

    private:
        void forwardGpu(const float* const* inputs, float* output, cudaStream_t stream, int batchSize = 1);

        nvinfer1::DataType m_datatype;

        int m_pad_size;
        int m_kernel_size;
        int m_max_displacement;
        int m_stride1;
        int m_stride2;
        int m_corr_type_multiply;

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

        nvinfer1::IPluginV2IOExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override;

        nvinfer1::IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

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
};

REGISTER_TENSORRT_PLUGIN(CorrelationPluginCreator);