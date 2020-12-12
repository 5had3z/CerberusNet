#include "cerberus.hpp"
#include "utils.hpp"

#include "../trt_plugins/correlation.hpp"
#include "../trt_plugins/grid_sampler.hpp"
#include "../trt_plugins/scatter_nd.hpp"

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

template<typename scalar_t>
void flow_image(const scalar_t* flow_image, u_char* rgb_image,
    size_t image_size, cudaStream_t Stream);

static Logger gLogger;
#define MAX_WORKSPACE (1UL << 34)

CERBERUS::CERBERUS() :
    m_last_batchsize(0),
    m_Network(nullptr),
    m_Builder(nullptr),
    m_ModelStream(nullptr),
    m_Engine(nullptr),
    m_Context(nullptr),
    m_CudaStream(nullptr),
    m_InputBuffer(nullptr)
{
    constexpr std::string_view labels_path { "/home/bryce/cs_labels.txt" };
    constexpr std::string_view ONNX_path { "/home/bryce/OCRNetSFD_dyn.onnx" };

    m_class_names = loadListFromTextFile(std::string{labels_path});

    //Engine Loading Things here 
    std::string engine_path{ONNX_path.substr(0, ONNX_path.size()-5)};
    engine_path += "_"+m_Precision+".trt";

    if (fileExists(engine_path, true)) {
        loadSerializedEngine(engine_path);
    } else {
        buildEngineFromONNX(ONNX_path, 2);
        writeSerializedEngine(engine_path);
    }

    m_Context = m_Engine->createExecutionContext();
    m_Context->setOptimizationProfile(0);
    m_max_batchsize = m_Engine->getMaxBatchSize();
    assert(m_Context != nullptr);

    auto volume = [](const nvinfer1::Dims& d){ 
        return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
    };

    for (int b = 0; b < m_Engine->getNbBindings(); ++b)
    {
        nvinfer1::Dims binding_dims = m_Engine->getBindingDimensions(b);
        binding_dims.d[0] = m_max_batchsize;
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

    NV_CUDA_CHECK(cudaStreamCreate(&m_CudaStream));
    this->allocateBuffers();

    NV_CUDA_CHECK(cudaMalloc(&m_dev_class_colourmap, m_class_colourmap.size()));
    NV_CUDA_CHECK(cudaMemcpy(m_dev_class_colourmap, m_class_colourmap.data(),
        m_class_colourmap.size(), cudaMemcpyHostToDevice));
}

CERBERUS::~CERBERUS()
{
    NV_CUDA_CHECK(cudaStreamDestroy(m_CudaStream));
    NV_CUDA_CHECK(cudaFree(m_dev_class_colourmap));

    if(m_InputBuffer) {
        NV_CUDA_CHECK(cudaFree(m_InputBuffer));
    }

    if (m_Context) {
        m_Context->destroy();
        m_Context = nullptr;
    }

    if (m_Engine) {
        m_Engine->destroy();
        m_Engine = nullptr;
    }
}

void CERBERUS::buildEngineFromONNX(const std::string_view onnx_path, size_t max_batchsize)
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
    m_Builder->setMaxBatchSize(max_batchsize);

    if (m_Precision == "FP16"){
        assert((m_Builder->platformHasFastFp16()) && "Platform does not support FP16");
        netcfg->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "Precision mode: FP16" << std::endl;
    }
    else if(m_Precision =="FP32") {
        std::cout << "Precision mode: FP32" << std::endl;
    }
    else {
        assert((false) && "Unsupported precision type");
    }

    // Add optimisation profiles for Network
    auto profile = m_Builder->createOptimizationProfile();
    for (int b = 0; b < m_Network->getNbInputs(); ++b)
    {
        auto binding_dims = m_Network->getInput(b)->getDimensions();
        auto binding_name = m_Network->getInput(b)->getName();
        binding_dims.d[0] = 1;
        profile->setDimensions(binding_name, nvinfer1::OptProfileSelector::kMIN, binding_dims);
        profile->setDimensions(binding_name, nvinfer1::OptProfileSelector::kOPT, binding_dims);
        binding_dims.d[0] = max_batchsize;
        profile->setDimensions(binding_name, nvinfer1::OptProfileSelector::kMAX, binding_dims);
    }
    netcfg->addOptimizationProfile(profile);

    m_Engine = m_Builder->buildEngineWithConfig(*m_Network, *netcfg);
    assert(m_Engine);

    std::cout << "Engine Built" << std::endl;

    /* we can clean up our mess */
    netcfg->destroy();
    parser->destroy();
    m_Network->destroy();
    m_Builder->destroy();
}

void CERBERUS::writeSerializedEngine(const std::string& engine_path)
{
    std::cout << "Serializing TensorRT Engine..." << std::endl;
    assert(m_Engine && "Invalid TensorRT Engine");
    m_ModelStream = m_Engine->serialize();
    assert(m_ModelStream && "Unable to serialize engine");
    assert(!engine_path.empty() && "Unable to save, No engine path");

    // Write engine to output file
    std::stringstream trtModelStream;
    trtModelStream.seekg(0, trtModelStream.beg);
    trtModelStream.write(static_cast<const char*>(m_ModelStream->data()), m_ModelStream->size());
    std::ofstream outFile;
    outFile.open(engine_path);
    outFile << trtModelStream.rdbuf();
    outFile.close();

    std::cout << "Serialized plan file cached at location : " << engine_path << std::endl;
}

void CERBERUS::loadSerializedEngine(const std::string& engine_path)
{
    // Reading the model in memory
    std::cout << "Loading TRT Engine: " << engine_path.c_str() << std::endl;
    assert(fileExists(engine_path, true));
    std::stringstream trtModelStream;
    trtModelStream.seekg(0, trtModelStream.beg);
    std::ifstream cache(engine_path);
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
    for (size_t i=0; i<m_max_batchsize; ++i)
    {
        // Allocating GPU memory for input and output tensors
        TRT_Buffer trt_buffer;
        trt_buffer.resize(m_Engine->getNbBindings());

        for (auto& tensor : m_InputTensors)
        {
            NV_CUDA_CHECK(trt_buffer.allocate_memory(
                tensor.bindingIndex, tensor.volume * sizeof(float)));
        }

        if (m_SegmentationTensor.bindingIndex != -1) {
            NV_CUDA_CHECK(trt_buffer.allocate_memory(
                m_SegmentationTensor.bindingIndex, m_SegmentationTensor.volume * sizeof(float)));
        }

        if (m_FlowTensor.bindingIndex != -1) {
            NV_CUDA_CHECK(trt_buffer.allocate_memory(
                m_FlowTensor.bindingIndex, m_FlowTensor.volume * sizeof(float)));
        }

        if (m_DepthTensor.bindingIndex != -1) {
            NV_CUDA_CHECK(trt_buffer.allocate_memory(
                m_DepthTensor.bindingIndex, m_DepthTensor.volume * sizeof(float)));
        }

        m_TRT_buffers.emplace_back(std::move(trt_buffer));
    }
}

cv::Mat CERBERUS::get_seg_class(size_t batch_indx) const
{
    if (batch_indx > m_max_batchsize) {
        throw std::runtime_error{"Batch indx requested exceeds maximum batchsize"};
    }
    else if (batch_indx > m_last_batchsize) {
        std::cerr << "Batch indx requested exceeds last batchsize\n";
    }

    cudaEventSynchronize(m_TRT_buffers[batch_indx].infer_status);
    cudaStream_t stream;
    NV_CUDA_CHECK(cudaStreamCreate(&stream));

    cv::Mat seg_argmax(cv::Size(m_InputW, m_InputH), CV_8UC1);
    void* gpu_buffer;
    cudaMalloc(&gpu_buffer, seg_argmax.total()*seg_argmax.elemSize1());
    argmax_chw((float*)m_TRT_buffers[batch_indx].at(m_SegmentationTensor.bindingIndex),
        (uchar*)gpu_buffer, 19, m_InputW * m_InputH, stream);
    cudaMemcpy(seg_argmax.data, gpu_buffer, seg_argmax.total()*seg_argmax.elemSize1(), cudaMemcpyDeviceToHost);
    cudaFree(gpu_buffer);
    cudaStreamDestroy(stream);
    return seg_argmax;
}

cv::Mat CERBERUS::get_seg_image(size_t batch_indx) const
{
    if (batch_indx > m_max_batchsize) {
        throw std::runtime_error{"Batch indx requested exceeds maximum batchsize"};
    }
    else if (batch_indx > m_last_batchsize) {
        std::cerr << "Batch indx requested exceeds last batchsize\n";
    }

    NV_CUDA_CHECK(cudaEventSynchronize(m_TRT_buffers[batch_indx].infer_status));

    const size_t n_px = m_InputW*m_InputH;
    cudaStream_t stream;
    NV_CUDA_CHECK(cudaStreamCreate(&stream));

    // Allocate temporary spaces for argmax and rgb image on GPU
    void* seg_argmax;
    void* seg_colour;
    NV_CUDA_CHECK(cudaMalloc(&seg_argmax, n_px*sizeof(uchar)));
    NV_CUDA_CHECK(cudaMalloc(&seg_colour, 3U*n_px*sizeof(uchar)));

    argmax_chw((float*)m_TRT_buffers[batch_indx].at(m_SegmentationTensor.bindingIndex),
        (uchar*)seg_argmax, 19, n_px, stream);

    seg_image<uchar, 19>((uchar*)seg_argmax, (uchar*)seg_colour,
        (uchar*)m_dev_class_colourmap, n_px, stream);
    cudaStreamSynchronize(stream);
    NV_CUDA_CHECK(cudaFree(seg_argmax));

    cv::Mat seg_colour_mat(cv::Size(m_InputW, m_InputH), CV_8UC3);
    NV_CUDA_CHECK(cudaMemcpy(seg_colour_mat.data, seg_colour, 3U*n_px*sizeof(uchar), cudaMemcpyDeviceToHost));
    NV_CUDA_CHECK(cudaFree(seg_colour));
    NV_CUDA_CHECK(cudaStreamDestroy(stream));

    return seg_colour_mat;
}

cv::Mat CERBERUS::get_depth(size_t batch_indx) const
{
    if (batch_indx > m_max_batchsize) {
        throw std::runtime_error{"Batch indx requested exceeds maximum batchsize"};
    }
    else if (batch_indx > m_last_batchsize) {
        std::cerr << "Batch indx requested exceeds last batchsize\n";
    }

    NV_CUDA_CHECK(cudaEventSynchronize(m_TRT_buffers[batch_indx].infer_status));

    cv::Mat depth_image(cv::Size(m_InputW, m_InputH), CV_32FC1);
    cudaMemcpy(depth_image.data, m_TRT_buffers[batch_indx].at(m_DepthTensor.bindingIndex),
        m_DepthTensor.volume / m_max_batchsize * sizeof(float), cudaMemcpyDeviceToHost);
    return depth_image / 80.f;
}

cv::Mat CERBERUS::get_flow(size_t batch_indx) const
{
    if (batch_indx > m_max_batchsize) {
        throw std::runtime_error{"Batch indx requested exceeds maximum batchsize"};
    }
    else if (batch_indx > m_last_batchsize) {
        std::cerr << "Batch indx requested exceeds last batchsize\n";
    }
    NV_CUDA_CHECK(cudaEventSynchronize(m_TRT_buffers[batch_indx].infer_status));

    const size_t n_px = m_InputW*m_InputH;
    
    void* dev_flow_image;
    NV_CUDA_CHECK(cudaMalloc(&dev_flow_image, 3U*n_px*sizeof(uchar)));

    cudaStream_t stream;
    NV_CUDA_CHECK(cudaStreamCreate(&stream));
    flow_image((float*)m_TRT_buffers[batch_indx].at(m_FlowTensor.bindingIndex), (uchar*)dev_flow_image, n_px, stream);
    NV_CUDA_CHECK(cudaStreamSynchronize(stream));
    NV_CUDA_CHECK(cudaStreamDestroy(stream));

    cv::Mat flow_image(cv::Size(m_InputW, m_InputH), CV_8UC3);
    NV_CUDA_CHECK(cudaMemcpy(flow_image.data, dev_flow_image, 3U*n_px*sizeof(uchar), cudaMemcpyDeviceToHost));
    NV_CUDA_CHECK(cudaFree(dev_flow_image));
    return flow_image;
}

template <typename MatType>
void CERBERUS::image_pair_inference(const std::pair<MatType, MatType> &img_sequence)
{
    m_last_batchsize = 1;
    this->allocate_image_pair(img_sequence);
}

template void CERBERUS::image_pair_inference(const std::pair<cv::Mat, cv::Mat> &img_sequence);
template void CERBERUS::image_pair_inference(const std::pair<cv::cuda::GpuMat, cv::cuda::GpuMat> &img_sequence);

template <typename MatType>
void CERBERUS::image_pair_inference(const std::vector<std::pair<MatType, MatType>> &img_sequence_vector)
{
    if (img_sequence_vector.empty()) {
        std::cerr << "Vector of images for inference is empty\n"; return;
    } else {
        m_last_batchsize = img_sequence_vector.size();
    }

    for (const auto& img_sequence : img_sequence_vector) {
        this->allocate_image_pair(img_sequence);
    }
}

template void CERBERUS::image_pair_inference(const std::vector<std::pair<cv::Mat, cv::Mat>> &img_sequence_vector);
template void CERBERUS::image_pair_inference(const std::vector<std::pair<cv::cuda::GpuMat, cv::cuda::GpuMat>> &img_sequence_vector);

template <typename MatType>
void CERBERUS::allocate_image_pair(const std::pair<MatType, MatType> &img_sequence)
{
    auto available_buffer = std::find_if(m_TRT_buffers.begin(), m_TRT_buffers.end(),
        [](const TRT_Buffer& buffer){ return cudaEventQuery(buffer.infer_status) == cudaSuccess; });

    if (available_buffer != m_TRT_buffers.end()) {
        cvmat_to_input_buffer(img_sequence.first, 0, *available_buffer);
        cvmat_to_input_buffer(img_sequence.second, 1, *available_buffer);
        if(!m_Context->enqueueV2(available_buffer->data(), available_buffer->stream, &available_buffer->infer_status)) {
            std::cerr << "TRT Enqueue Fail\n";
        }
        std::cout << cudaEventQuery(available_buffer->infer_status) << std::endl;
    }
    else {
        std::cerr << "All buffers occupied\n";
    }
}

template<typename MatType>
void CERBERUS::cvmat_to_input_buffer(const MatType &img, size_t input_indx, TRT_Buffer& trt_buffer)
{
    if (img.rows != m_InputH || img.cols != m_InputW || img.channels() != m_InputC) {
        throw std::runtime_error{ "Input image dimensions are different from engine input" };
    }
    // CUDA kernel to reshape the non-continuous GPU Mat structure and make it channel-first continuous
    if constexpr(std::is_same<MatType, cv::Mat>::value) {
        if (!m_InputBuffer) { NV_CUDA_CHECK(cudaMalloc(&m_InputBuffer, this->getInputVolume())); }
        cudaMemcpyAsync(m_InputBuffer, img.data, this->getInputVolume(), cudaMemcpyHostToDevice, trt_buffer.stream);
    }
    else if constexpr(std::is_same<MatType, cv::cuda::GpuMat>::value) {
        m_InputBuffer = img.data;
    }
    else {
        throw std::runtime_error{"INCOMPATIBLE INPUT FORMAT"};
    }

    float* trt_input = (float*)trt_buffer.at(m_InputTensors[input_indx].bindingIndex);

    nhwc2nchw((unsigned char*)m_InputBuffer, trt_input, 
        m_InputH*m_InputW, m_InputC, m_InputC*m_InputW,
        static_cast<int>(img.step / img.elemSize1()), trt_buffer.stream);
    
    normalize_image_chw(trt_input, m_InputH * m_InputW, mean, std, trt_buffer.stream);

    NV_CUDA_CHECK(cudaGetLastError());
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