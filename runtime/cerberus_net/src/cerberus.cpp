#include "cerberus.hpp"
#include "utils.hpp"
#include "../trt_plugins/trt_utils.hpp"

#include <numeric>
#include <iostream>

#include <NvOnnxParser.h>

static Logger gLogger;
#define MAX_WORKSPACE (1 << 30)

CERBERUS::CERBERUS() :
    m_Network(nullptr),
    m_Builder(nullptr),
    m_ModelStream(nullptr),
    m_Engine(nullptr),
    m_Context(nullptr),
    m_CudaStream(nullptr)
{
    constexpr std::string_view labels_path { "/config/cone3_labels.txt" };
    constexpr std::string_view ONNX_path { "/engine/peleenet_cropped.onnx" };

    m_class_names = loadListFromTextFile(std::string{labels_path});

     //Engine Loading Things here 
    if (m_EnginePath.empty()){
        m_EnginePath = ONNX_path.substr(0, ONNX_path.size()-5);
        m_EnginePath += "_"+m_Precision+".trt";
    }

    if (fileExists(m_EnginePath, true)) {
        loadSerializedEngine();
    } else {
        buildEngineFromONNX(ONNX_path);
        writeSerializedEngine();
    }

    m_Context = m_Engine->createExecutionContext();
    assert(m_Context != nullptr);

    auto volume = [](const nvinfer1::Dims& d){ 
        return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
    };

    m_InputTensors.reserve(2);
    // IDX:0-1 (inputs), IDX:2-5 (outputs)
    for (int b = 0; b < m_Engine->getNbBindings(); ++b)
    {
        const nvinfer1::Dims binding_dims = m_Engine->getBindingDimensions(b);
        if (m_Engine->bindingIsInput(b)){ 
            TensorInfo new_tensor;
            new_tensor.volume = volume(binding_dims);
            new_tensor.blobName = m_Engine->getBindingName(b);
            new_tensor.bindingIndex = b;
            m_InputTensors.push_back(new_tensor);
            m_InputH = binding_dims.d[1];
            m_InputW = binding_dims.d[2];
        }
        // 19 Channel outputs means segmentation
        else if (binding_dims.d[0] == 19){
            m_SegmentationTensor.volume = volume(binding_dims);
            m_SegmentationTensor.blobName = m_Engine->getBindingName(b);
            m_SegmentationTensor.bindingIndex = b;
        } 
        // 2 Channel outputs means flow
        else if (binding_dims.d[0] == 2){
            m_FlowTensor.volume = volume(binding_dims);
            m_FlowTensor.blobName = m_Engine->getBindingName(b);
            m_FlowTensor.bindingIndex = b;
        } 
        // 1 Channel output means depth
        else if (binding_dims.d[0] == 1){
            m_DepthTensor.volume = volume(binding_dims);
            m_DepthTensor.blobName = m_Engine->getBindingName(b);
            m_DepthTensor.bindingIndex = b;
        }
        else {
            std::cerr << "Invalid tensor channels: " << binding_dims.d[0] << "\n";
        }
    }

    cudaStreamCreate(&m_CudaStream);
    allocateBuffers();
}

CERBERUS::~CERBERUS()
{
    for (auto& tensor : m_InputTensors)
    {
        NV_CUDA_CHECK(cudaFree(&m_DeviceBuffers.at(tensor.bindingIndex)));
    }
    NV_CUDA_CHECK(cudaFree(&m_DeviceBuffers.at(m_SegmentationTensor.bindingIndex)));
    NV_CUDA_CHECK(cudaFree(&m_DeviceBuffers.at(m_FlowTensor.bindingIndex)));
    NV_CUDA_CHECK(cudaFree(&m_DeviceBuffers.at(m_DepthTensor.bindingIndex)));
}

void CERBERUS::buildEngineFromONNX(const std::string_view onnx_path)
{
    m_Builder = nvinfer1::createInferBuilder(gLogger);
    const auto netflags = 1 << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    m_Network = m_Builder->createNetworkV2(netflags);
    
    auto parser = nvonnxparser::createParser(*m_Network, gLogger);
    int verbosity = (int) nvinfer1::ILogger::Severity::kINFO;

    std::cout << "Parsing ..." << onnx_path << std::endl;
    if (!parser->parseFromFile(onnx_path.begin(), verbosity))
    {
       std::cout << "failed";
    }
    std::cout << "Parsing done" << std::endl;

    // ----------------------------------
    /* we create the engine */
    m_Builder->setMaxBatchSize(m_maxBatchSize);
    m_Builder->setMaxWorkspaceSize(MAX_WORKSPACE);

    m_Engine = m_Builder->buildCudaEngine(*m_Network);
    assert(m_Engine);    

    /* we can clean up our mess */
    parser->destroy();
    m_Network->destroy();
    m_Builder->destroy();
}

void CERBERUS::writeSerializedEngine()
{
}

void CERBERUS::loadSerializedEngine()
{
}

void CERBERUS::allocateBuffers()
{
    // Allocating GPU memory for input and output tensors
    m_DeviceBuffers.resize(m_Engine->getNbBindings(), nullptr);

    for (auto& tensor : m_InputTensors)
    {
        NV_CUDA_CHECK(cudaMallocManaged(&m_DeviceBuffers.at(tensor.bindingIndex),
            m_maxBatchSize * 3U * m_InputW * m_InputH * sizeof(float)));
    }
    NV_CUDA_CHECK(cudaMallocManaged(&m_DeviceBuffers.at(m_SegmentationTensor.bindingIndex),
            m_maxBatchSize * 19U * m_InputW * m_InputH * sizeof(float)));

    NV_CUDA_CHECK(cudaMallocManaged(&m_DeviceBuffers.at(m_FlowTensor.bindingIndex),
            m_maxBatchSize * 2U * m_InputW * m_InputH * sizeof(float)));

    NV_CUDA_CHECK(cudaMallocManaged(&m_DeviceBuffers.at(m_DepthTensor.bindingIndex),
            m_maxBatchSize * 1U * m_InputW * m_InputH * sizeof(float)));
}

void CERBERUS::doInference(const cv::cuda::GpuMat &img, const cv::cuda::GpuMat &img_seq)
{
    cudaMemcpy(img.data, m_DeviceBuffers[0], m_InputW * m_InputH * 3U * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(img_seq.data, m_DeviceBuffers[1], m_InputW * m_InputH * 3U * sizeof(float), cudaMemcpyDeviceToDevice);
    m_Context->enqueueV2(m_DeviceBuffers.data(), m_CudaStream, nullptr);
}

void CERBERUS::doInference(const cv::Mat &img, const cv::Mat &img_seq)
{
    cudaMemcpy(img.data, m_DeviceBuffers[0], m_InputW * m_InputH * 3U * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(img_seq.data, m_DeviceBuffers[1], m_InputW * m_InputH * 3U * sizeof(float), cudaMemcpyHostToDevice);
    m_Context->enqueueV2(m_DeviceBuffers.data(), m_CudaStream, nullptr);
}

void Logger::log(Severity severity, const char* msg)
{
    // suppress messages with severity enum value greater than the reportable
    if (severity > reportableSeverity)
        return;

    switch (severity)
    {
    case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
    case Severity::kERROR: std::cerr << "ERROR: "; break;
    case Severity::kWARNING: std::cerr << "WARNING: "; break;
    case Severity::kINFO: std::cerr << "INFO: "; break;
    default: std::cerr << "UNKNOWN: "; break;
    }
    std::cerr << msg << std::endl;
}