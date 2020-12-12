#pragma once

#include <utility>
#include <algorithm>
#include <iostream>
#include <string_view>
#include <type_traits>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#define NV_CUDA_CHECK(status)                                                                      \
    {                                                                                              \
        if (status != 0)                                                                           \
        {                                                                                          \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) << " in file " << __FILE__ \
                      << " at line " << __LINE__ << std::endl;                                     \
            abort();                                                                               \
        }                                                                                          \
    }

// Declare CUDA Kernels
void nhwc2nchw(const unsigned char* source, float* dest, int channelSize, int channelsNum, int rowElements, int rowSize, cudaStream_t Stream);
template<typename scalar_t, size_t n_ch>
void normalize_image_chw(scalar_t* image, size_t ch_stride, const std::array<scalar_t, n_ch> &mean,
    const std::array<scalar_t, n_ch> &std, cudaStream_t Stream);

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
public:
    //  Change this to kWARNING if you don't want to see all garbage when building the engine
    Logger(Severity severity = Severity::kINFO)
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
};

struct TRT_Buffer
{
public:
    cudaEvent_t infer_status;
    cudaStream_t stream;
    std::vector<void*> device_buffers;
    std::vector<size_t> buffer_sizes;

    explicit TRT_Buffer()
    {
        NV_CUDA_CHECK(cudaStreamCreate(&stream));
        NV_CUDA_CHECK(cudaEventCreate(&infer_status));
    }

    explicit TRT_Buffer(const TRT_Buffer& obj)
    {
        NV_CUDA_CHECK(cudaStreamCreate(&stream));
        NV_CUDA_CHECK(cudaEventCreate(&infer_status));
        for (auto buffer_size : obj.buffer_sizes)
        {
            device_buffers.emplace_back(nullptr);
            buffer_sizes.emplace_back(buffer_size);
            NV_CUDA_CHECK(cudaMallocManaged(&device_buffers.back(), buffer_size));
            NV_CUDA_CHECK(cudaMemcpy(device_buffers.back(),
                obj.device_buffers[device_buffers.size()-1], buffer_size, cudaMemcpyDefault));
        }
    }

    ~TRT_Buffer()
    {
        NV_CUDA_CHECK(cudaStreamDestroy(stream));
        NV_CUDA_CHECK(cudaEventDestroy(infer_status));
        for (auto buffer : device_buffers){
            if (buffer) { NV_CUDA_CHECK(cudaFree(buffer)); }
        }
    }

    cudaError_t allocate_memory(size_t buffer_indx, size_t size) noexcept
    {
        assert(buffer_indx < device_buffers.size());
        buffer_sizes[buffer_indx] = size;
        return cudaMallocManaged(&device_buffers[buffer_indx], size);
    }

    void** data() noexcept
    {
        return device_buffers.data();
    }

    void* operator[] (size_t buffer_indx) const noexcept
    {
        return device_buffers[buffer_indx];
    }

    void* at(size_t buffer_indx) const noexcept
    {
        return device_buffers[buffer_indx];
    }

    void resize(size_t size)
    {
        device_buffers.resize(size, nullptr);
        buffer_sizes.resize(size, 0);
    }

    void reserve(size_t size)
    {
        device_buffers.reserve(size);
        buffer_sizes.resize(size);
    }    
};

class CERBERUS
{
public:
    CERBERUS();
    virtual ~CERBERUS();

    template <typename MatType>
    void image_pair_inference(const std::pair<MatType, MatType> &img_sequence);

    template <typename MatType>
    void image_pair_inference(const std::vector<std::pair<MatType, MatType>> &img_sequence_vector);

    [[nodiscard]] cv::Mat get_seg_class(size_t batch_indx = 0) const;
    [[nodiscard]] cv::Mat get_seg_image(size_t batch_indx = 0) const;
    [[nodiscard]] cv::Mat get_depth(size_t batch_indx = 0) const;
    [[nodiscard]] cv::Mat get_flow(size_t batch_indx = 0) const;

    [[nodiscard]] std::string getClassName(const int& classID) const noexcept { return m_class_names[classID]; }
    [[nodiscard]] std::size_t getNumClasses() const noexcept { return m_class_names.size(); }

    [[nodiscard]] std::size_t getInputC() const noexcept { return m_InputC; }
    [[nodiscard]] std::size_t getInputH() const noexcept { return m_InputH; }
    [[nodiscard]] std::size_t getInputW() const noexcept { return m_InputW; }
    [[nodiscard]] std::size_t getInputVolume() const noexcept { return m_InputC * m_InputH * m_InputW; }

private:
    size_t m_max_batchsize;
    size_t m_last_batchsize;
    size_t m_InputW;
    size_t m_InputH;
    size_t m_InputC;

    std::vector<std::string> m_class_names;
    static constexpr auto m_class_colourmap = 
        std::array<uchar, 19*3>{
            128, 64,128,
            244, 35,232,
            70, 70, 70,
            102,102,156,
            190,153,153,
            153,153,153,
            250,170, 30,
            220,220,  0,
            107,142, 35,
            152,251,152,
             70,130,180,
            220, 20, 60,
            255,  0,  0,
              0,  0,142,
              0,  0, 70,
              0, 60,100,
              0, 80,100,
              0,  0,230,
            119, 11, 32};
    void* m_dev_class_colourmap;

    const std::string m_Precision = PRECISION;// Defined in cmake

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
    std::vector<TRT_Buffer> m_TRT_buffers;
    void* m_InputBuffer;

    static constexpr auto mean = std::array{0.485f, 0.456f, 0.406f};
    static constexpr auto std = std::array{0.229f, 0.224f, 0.225f};

    void buildEngineFromONNX(const std::string_view onnx_path, size_t batchsize);
    void writeSerializedEngine(const std::string& engine_path);
    void loadSerializedEngine(const std::string& engine_path);
    void allocateBuffers();

    template<typename MatType>
    void cvmat_to_input_buffer(const MatType &img, size_t input_indx, TRT_Buffer& trt_buffer);

    template<typename MatType>
    void allocate_image_pair(const std::pair<MatType, MatType> &img_sequence);
};
