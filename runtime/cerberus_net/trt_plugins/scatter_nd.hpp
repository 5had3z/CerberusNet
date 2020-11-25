#pragma once

#include <cuda_runtime.h>

#include <vector>
#include <string>

#include <NvInfer.h>

class ScatterNDPlugin: public nvinfer1::IPluginV2IOExt
{
    public:
        ScatterNDPlugin(const nvinfer1::PluginFieldCollection& fc);

        ScatterNDPlugin(const void* data, size_t length);

        ~ScatterNDPlugin() override = default;

        int getNbOutputs() const override { return 1; }

        nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override;

        int initialize() override;

        void terminate() override;

        size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }

        int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

        size_t getSerializationSize() const override;

        void serialize(void* buffer) const override;

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
        nvinfer1::DataType m_datatype;
        const char* mPluginNamespace;
};


class ScatterNDPluginCreator : public nvinfer1::IPluginCreator
{
    public:
        ScatterNDPluginCreator();

        ~ScatterNDPluginCreator() override = default;

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
        std::string mPluginName;
};

REGISTER_TENSORRT_PLUGIN(ScatterNDPluginCreator);