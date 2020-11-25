#include "cerberus.hpp"
#include "utils.hpp"

#include <fstream>
#include <numeric>
#include <iostream>

#include <NvOnnxParser.h>

template<typename scalar_t, typename intergral_t>
void argmax_chw(const scalar_t* input, intergral_t* output,
    size_t n_classes, size_t ch_stride, cudaStream_t Stream);

template<typename intergral_t, size_t n_classes>
void seg_image(const intergral_t* input, u_char* output, const u_char* colour_map,
    size_t image_size, cudaStream_t Stream);

static Logger gLogger;
#define MAX_WORKSPACE (1UL << 32)

CERBERUS::CERBERUS() :
    m_Network(nullptr),
    m_Builder(nullptr),
    m_ModelStream(nullptr),
    m_Engine(nullptr),
    m_Context(nullptr),
    m_CudaStream(nullptr),
    m_InputBuffer(nullptr)
{
    constexpr std::string_view labels_path { "/home/bryce/cs_labels.txt" };
    constexpr std::string_view ONNX_path { "/home/bryce/export_test.onnx" };

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

    for (int b = 0; b < m_Engine->getNbBindings(); ++b)
    {
        const nvinfer1::Dims binding_dims = m_Engine->getBindingDimensions(b);
        if (m_Engine->bindingIsInput(b)){ 
            TensorInfo new_tensor;
            new_tensor.volume = volume(binding_dims);
            new_tensor.blobName = m_Engine->getBindingName(b);
            new_tensor.bindingIndex = b;
            m_InputTensors.push_back(new_tensor);
            m_InputC = binding_dims.d[1];
            m_InputH = binding_dims.d[2];
            m_InputW = binding_dims.d[3];
        }
        // 19 Channel outputs means segmentation
        else if (binding_dims.d[1] == 19){
            m_SegmentationTensor.volume = volume(binding_dims);
            m_SegmentationTensor.blobName = m_Engine->getBindingName(b);
            m_SegmentationTensor.bindingIndex = b;
        } 
        // 2 Channel outputs means flow
        else if (binding_dims.d[1] == 2){
            m_FlowTensor.volume = volume(binding_dims);
            m_FlowTensor.blobName = m_Engine->getBindingName(b);
            m_FlowTensor.bindingIndex = b;
        } 
        // 1 Channel output means depth
        else if (binding_dims.d[1] == 1){
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

    cudaMalloc(&m_dev_class_colourmap, m_class_colourmap.size());
    cudaMemcpy(m_dev_class_colourmap, m_class_colourmap.data(), m_class_colourmap.size(), cudaMemcpyHostToDevice);
}

CERBERUS::~CERBERUS()
{
    for (auto& tensor : m_InputTensors)
    {
        NV_CUDA_CHECK(cudaFree(m_DeviceBuffers.at(tensor.bindingIndex)));
    }

    if (m_SegmentationTensor.bindingIndex != -1){
        NV_CUDA_CHECK(cudaFree(m_DeviceBuffers.at(m_SegmentationTensor.bindingIndex)));
    }
    if (m_FlowTensor.bindingIndex != -1){
        NV_CUDA_CHECK(cudaFree(m_DeviceBuffers.at(m_FlowTensor.bindingIndex)));
    }
    if (m_DepthTensor.bindingIndex != -1){
        NV_CUDA_CHECK(cudaFree(m_DeviceBuffers.at(m_DepthTensor.bindingIndex)));
    }

    NV_CUDA_CHECK(cudaStreamDestroy(m_CudaStream));
    NV_CUDA_CHECK(cudaFree(m_dev_class_colourmap));

    if (m_Context) {
        m_Context->destroy();
        m_Context = nullptr;
    }

    if (m_Engine) {
        m_Engine->destroy();
        m_Engine = nullptr;
    }
}

void CERBERUS::buildEngineFromONNX(const std::string_view onnx_path)
{
    m_Builder = nvinfer1::createInferBuilder(gLogger);
    const auto netflags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    m_Network = m_Builder->createNetworkV2(netflags);
    
    auto parser = nvonnxparser::createParser(*m_Network, gLogger);
    int verbosity = (int) nvinfer1::ILogger::Severity::kWARNING;

    std::cout << "Parsing " << onnx_path << std::endl;
    if (!parser->parseFromFile(onnx_path.begin(), verbosity))
    {
       std::cout << "ONNX Parsing Failed";
    }
    std::cout << "ONNX Parsing Done" << std::endl;

    nvinfer1::IBuilderConfig* netcfg = m_Builder->createBuilderConfig();
    netcfg->setMaxWorkspaceSize(MAX_WORKSPACE);
    m_Builder->setMaxBatchSize(m_maxBatchSize);

    m_Engine = m_Builder->buildEngineWithConfig(*m_Network, *netcfg);
    assert(m_Engine);

    std::cout << "Engine Built" << std::endl;

    /* we can clean up our mess */
    netcfg->destroy();
    parser->destroy();
    m_Network->destroy();
    m_Builder->destroy();
}

void CERBERUS::writeSerializedEngine()
{
    std::cout << "Serializing TensorRT Engine..." << std::endl;
    assert(m_Engine && "Invalid TensorRT Engine");
    m_ModelStream = m_Engine->serialize();
    assert(m_ModelStream && "Unable to serialize engine");
    assert(!m_EnginePath.empty() && "Unable to save, No engine path");

    // Write engine to output file
    std::stringstream trtModelStream;
    trtModelStream.seekg(0, trtModelStream.beg);
    trtModelStream.write(static_cast<const char*>(m_ModelStream->data()), m_ModelStream->size());
    std::ofstream outFile;
    outFile.open(m_EnginePath);
    outFile << trtModelStream.rdbuf();
    outFile.close();

    std::cout << "Serialized plan file cached at location : " << m_EnginePath << std::endl;
}

void CERBERUS::loadSerializedEngine()
{
    // Reading the model in memory
    std::cout << "Loading TRT Engine: " << m_EnginePath.c_str() << std::endl;
    assert(fileExists(m_EnginePath, true));
    std::stringstream trtModelStream;
    trtModelStream.seekg(0, trtModelStream.beg);
    std::ifstream cache(m_EnginePath);
    assert(cache.good());
    trtModelStream << cache.rdbuf();
    cache.close();

    // Calculating model size
    trtModelStream.seekg(0, std::ios::end);
    const int modelSize = trtModelStream.tellg();
    trtModelStream.seekg(0, std::ios::beg);
    void* modelMem = malloc(modelSize);
    trtModelStream.read((char*) modelMem, modelSize);

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    m_Engine = runtime->deserializeCudaEngine(modelMem, modelSize);
    free(modelMem);
    runtime->destroy();
}

void CERBERUS::allocateBuffers()
{
    // Allocating GPU memory for input and output tensors
    m_DeviceBuffers.resize(m_Engine->getNbBindings(), nullptr);

    for (auto& tensor : m_InputTensors)
    {
        NV_CUDA_CHECK(cudaMallocManaged(&m_DeviceBuffers.at(tensor.bindingIndex),
            m_maxBatchSize * tensor.volume * sizeof(float)));
    }

    if (m_SegmentationTensor.bindingIndex != -1) {
        NV_CUDA_CHECK(cudaMallocManaged(&m_DeviceBuffers.at(m_SegmentationTensor.bindingIndex),
            m_maxBatchSize * m_SegmentationTensor.volume * sizeof(float)));
    }

    if (m_FlowTensor.bindingIndex != -1) {
        NV_CUDA_CHECK(cudaMallocManaged(&m_DeviceBuffers.at(m_FlowTensor.bindingIndex),
            m_maxBatchSize * m_FlowTensor.volume * sizeof(float)));
    }

    if (m_DepthTensor.bindingIndex != -1) {
        NV_CUDA_CHECK(cudaMallocManaged(&m_DeviceBuffers.at(m_DepthTensor.bindingIndex),
            m_maxBatchSize * m_DepthTensor.volume * sizeof(float)));
    }
}

cv::Mat CERBERUS::get_seg_class()
{
    cv::Mat seg_argmax(cv::Size(m_InputW, m_InputH), CV_8UC1);
    void* gpu_buffer;
    cudaMalloc(&gpu_buffer, seg_argmax.total()*seg_argmax.elemSize1());
    argmax_chw((float*)m_DeviceBuffers.at(m_SegmentationTensor.bindingIndex),
        (uchar*)gpu_buffer, 19, m_InputW * m_InputH, m_CudaStream);
    cudaMemcpy(seg_argmax.data, gpu_buffer, seg_argmax.total()*seg_argmax.elemSize1(), cudaMemcpyDeviceToHost);
    cudaFree(gpu_buffer);
    return seg_argmax;
}

cv::Mat CERBERUS::get_seg_image()
{
    const size_t n_px = m_InputW*m_InputH;

    // Allocate temporary spaces for argmax and rgb image on GPU
    void* seg_argmax;
    void* seg_colour;
    NV_CUDA_CHECK(cudaMalloc(&seg_argmax, n_px*sizeof(uchar)));
    NV_CUDA_CHECK(cudaMalloc(&seg_colour, 3U*n_px*sizeof(uchar)));

    argmax_chw((float*)m_DeviceBuffers.at(m_SegmentationTensor.bindingIndex),
        (uchar*)seg_argmax, 19, n_px, m_CudaStream);

    seg_image<uchar, 19>((uchar*)seg_argmax, (uchar*)seg_colour,
        (uchar*)m_dev_class_colourmap, n_px, m_CudaStream);
    NV_CUDA_CHECK(cudaFree(seg_argmax));

    cv::Mat seg_colour_mat(cv::Size(m_InputW, m_InputH), CV_8UC3);
    NV_CUDA_CHECK(cudaMemcpy(seg_colour_mat.data, seg_colour, 3U*n_px*sizeof(uchar), cudaMemcpyDeviceToHost));
    NV_CUDA_CHECK(cudaFree(seg_colour));

    return seg_colour_mat;
}

cv::Mat CERBERUS::get_depth()
{
    cv::Mat depth_image(cv::Size(m_InputW, m_InputH), CV_32FC1);
    cudaMemcpy(depth_image.data, m_DeviceBuffers.at(m_DepthTensor.bindingIndex),
        m_DepthTensor.volume * sizeof(float), cudaMemcpyDeviceToHost);
    return depth_image / 80.f;
}

cv::Mat CERBERUS::get_flow()
{

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
        case Severity::kVERBOSE: std::cerr << "VERBOSE: "; break;
        default: std::cerr << "UNKNOWN: "; break;
    }
    std::cerr << msg << std::endl;
}