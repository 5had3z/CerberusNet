#include "scatter_nd.hpp"
#include "trt_utils.hpp"

#include <cassert>
#include <cstring>

namespace
{
    const char* SCATTER_ND_PLUGIN_VERSION{"1"};
    const char* SCATTER_ND_PLUGIN_NAME{"ScatterND"};
} // namespace

nvinfer1::PluginFieldCollection ScatterNDPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> ScatterNDPluginCreator::mPluginAttributes;
