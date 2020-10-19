#pragma once

#include <string_view>

#include <opencv2/core/cuda.hpp>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
public:
    //  Change this to kWARNING if you don't want to see all garbage when building the engine
    Logger(Severity severity = Severity::kWARNING)
        : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char* msg) override;

    Severity reportableSeverity;
};

struct TensorInfo
{
    std::string blobName;
    uint64_t volume{0};
    int bindingIndex{-1};
    float* hostBuffer{nullptr}; // Float Ptr
};

class CERBERUS
{
public:
    CERBERUS();
    virtual ~CERBERUS();

    void doInference(const cv::cuda::GpuMat &img, const cv::cuda::GpuMat &img_seq);

    [[nodiscard]] std::string getClassName(const int& classID) const noexcept { return m_class_names[classID]; }
    [[nodiscard]] std::size_t getNumClasses(const int& classID) const noexcept { return m_class_names.size(); }

    [[nodiscard]] std::size_t getInputH() const noexcept { return m_InputH; }
    [[nodiscard]] std::size_t getInputW() const noexcept { return m_InputW; }

private:
    static constexpr uint m_maxBatchSize = 1;
    size_t m_InputW;
    size_t m_InputH;

    std::vector<std::string> m_class_names;

    const std::string m_Precision = PRECISION;// Defined in cmake
    std::string m_EnginePath; // Will be automatically inferred from ONNX_FILE

    nvinfer1::INetworkDefinition* m_Network;
    nvinfer1::IBuilder* m_Builder;
    nvinfer1::IHostMemory* m_ModelStream;
    nvinfer1::ICudaEngine* m_Engine;
    nvinfer1::IExecutionContext* m_Context;
    cudaStream_t m_CudaStream;

    std::vector<TensorInfo> m_InputTensors;
    TensorInfo m_FlowTensor;
    TensorInfo m_SegmentationTensor;
    TensorInfo m_DepthTensor;

    void buildEngineFromONNX(const std::string_view onnx_path);
    void writeSerializedEngine();
    void loadSerializedEngine();
    void allocateBuffers();
};
