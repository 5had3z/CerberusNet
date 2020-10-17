#include "cerberus.hpp"

#include <NvOnnxParser.h>

#define MAX_WORKSPACE (1 << 30)

CERBERUS::CERBERUS()
{
}

CERBERUS::~CERBERUS()
{
}

void CERBERUS::buildEngineFromONNX()
{
}

void CERBERUS::writeSerializedEngine()
{
}

void CERBERUS::loadSerializedEngine()
{
}

void CERBERUS::allocateBuffers()
{
}

void CERBERUS::doInference(const cv::cuda::GpuMat &img, const cv::cuda::GpuMat &img_seq)
{
}