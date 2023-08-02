/*
 * SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/*
 * This file is originally generated by NVIDIA-AI-IOT/tensorrt_plugin_generator,
 * GitHub repo: https://github.com/NVIDIA-AI-IOT/tensorrt_plugin_generator
 */

#ifndef DEBUG_PLUGIN
#define DEBUG_PLUGIN 1 // set debug mode, if you want to see the api call, set it to 1
#endif

#include "NvInfer.h"
#include "GridsampleIPluginV2DynamicExt.h"

#include <cassert>
#include <cstring>
#include <vector>
#include <iostream>

#if DEBUG_PLUGIN
#define DEBUG_LOG(...) {\
    std::cout << " ----> debug <---- call " << "[" << __FILE__ << ":" \
              << __LINE__ << "][" << __FUNCTION__ << "]" << std::endl;\
    }
#else
#define DEBUG_LOG(...)
#endif

using namespace nvinfer1;

namespace
{
const char* PLUGIN_VERSION{"1"};
const char* PLUGIN_NAME{"GridSample"};
} // namespace

// Static class fields initialization
PluginFieldCollection GridsampleIPluginV2DynamicExtCreator::mFC{};
std::vector<PluginField> GridsampleIPluginV2DynamicExtCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GridsampleIPluginV2DynamicExtCreator);

// Helper function for serializing plugin
template <typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template <typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}


std::string to_string(GridSamplerInterpolation interp) {
    switch(interp) {
        case GridSamplerInterpolation::Bilinear: return "Bilinear";
        case GridSamplerInterpolation::Nearest: return "Nearest";
        default: return "Unknown";
    }
}

std::string to_string(GridSamplerPadding padding) {
    switch(padding) {
        case GridSamplerPadding::Zeros: return "Zeros";
        case GridSamplerPadding::Border: return "Border";
        case GridSamplerPadding::Reflection: return "Reflection";
        default: return "Unknown";
    }
}

std::string to_string(GridSamplerDataType datatype) {
    switch(datatype) {
        case GridSamplerDataType::GFLOAT: return "float";
        case GridSamplerDataType::GHALF: return "half";
        default: return "Unknown";
    }
}

GridsampleIPluginV2DynamicExt::GridsampleIPluginV2DynamicExt(const std::string name, 
GridSamplerInterpolation interpolationMode, 
GridSamplerPadding paddingMode,
bool alignCorners)
    : mLayerName(name)
    , mInterpolationMode(interpolationMode)
    , mPaddingMode(paddingMode)
    , mAlignCorners(alignCorners)
{
    DEBUG_LOG();
}

// for clone
GridsampleIPluginV2DynamicExt::GridsampleIPluginV2DynamicExt(const std::string name, int inputChannel, int inputHeight,
    int inputWidth, int gridHeight, int gridWidth, GridSamplerInterpolation interpolationMode,
    GridSamplerPadding paddingMode, bool alignCorners, DataType type)
    : mLayerName(name)
    , mInputChannel(inputChannel)
    , mInputHeight(inputHeight)
    , mInputWidth(inputWidth)
    , mGridHeight(gridHeight)
    , mGridWidth(gridWidth)
    , mInterpolationMode(interpolationMode)
    , mPaddingMode(paddingMode)
    , mAlignCorners(alignCorners)
    , mType(type)
{
    DEBUG_LOG();
}

GridsampleIPluginV2DynamicExt::GridsampleIPluginV2DynamicExt(const std::string name, const void* serial_buf, size_t serial_size)
    : mLayerName(name)
{
    DEBUG_LOG();
    const char* d = reinterpret_cast<const char*>(serial_buf);
    const char* a = d;
    mInputChannel = readFromBuffer<size_t>(d);    
    mInputHeight = readFromBuffer<size_t>(d);    
    mInputWidth = readFromBuffer<size_t>(d);
    mGridHeight = readFromBuffer<size_t>(d);
    mGridWidth = readFromBuffer<size_t>(d);
    mInterpolationMode = readFromBuffer<GridSamplerInterpolation>(d);
    mPaddingMode = readFromBuffer<GridSamplerPadding>(d);
    mAlignCorners = readFromBuffer<bool>(d);
    mType = readFromBuffer<DataType>(d);
    assert(d == a + sizeof(size_t) * 5 + sizeof(GridSamplerInterpolation) + sizeof(GridSamplerPadding) + sizeof(bool) + sizeof(DataType));
}

GridsampleIPluginV2DynamicExt::~GridsampleIPluginV2DynamicExt() noexcept {}

// -------------------- IPluginV2 ----------------------

const char* GridsampleIPluginV2DynamicExt::getPluginType() const IS_NOEXCEPT
{
    DEBUG_LOG();
    return PLUGIN_NAME;
}

const char* GridsampleIPluginV2DynamicExt::getPluginVersion() const IS_NOEXCEPT
{
    DEBUG_LOG();
    return PLUGIN_VERSION;
}

int GridsampleIPluginV2DynamicExt::getNbOutputs() const IS_NOEXCEPT
{
    DEBUG_LOG();
    return 1;
}

// IMPORTANT: Memory allocated in the plug-in must be freed to ensure no memory leak.
// If resources are acquired in the initialize() function, they must be released in the terminate() function.
// All other memory allocations should be freed, preferably in the plug-in class destructor or in the destroy() method.

// Initialize the layer for execution.
// e.g. if the plugin require some extra device memory for execution. allocate in this function.
// for details please refer to
// https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v2.html
int GridsampleIPluginV2DynamicExt::initialize() IS_NOEXCEPT
{
    DEBUG_LOG();
    return 0;
}

void GridsampleIPluginV2DynamicExt::terminate() IS_NOEXCEPT
{
    DEBUG_LOG();
    // Release resources acquired during plugin layer initialization
}

size_t GridsampleIPluginV2DynamicExt::getSerializationSize() const IS_NOEXCEPT
{
    return sizeof(size_t) * 5 + sizeof(GridSamplerInterpolation) + sizeof(GridSamplerPadding) + sizeof(bool) + sizeof(DataType);
}

void GridsampleIPluginV2DynamicExt::serialize(void* buffer) const IS_NOEXCEPT
{
    DEBUG_LOG();
    char* d = reinterpret_cast<char*>(buffer);
    char* a = d;
    writeToBuffer<size_t>(d, mInputChannel);    
    writeToBuffer<size_t>(d, mInputHeight);    
    writeToBuffer<size_t>(d, mInputWidth);
    writeToBuffer<size_t>(d, mGridHeight);
    writeToBuffer<size_t>(d, mGridWidth);
    writeToBuffer<GridSamplerInterpolation>(d, mInterpolationMode);
    writeToBuffer<GridSamplerPadding>(d, mPaddingMode);
    writeToBuffer<bool>(d, mAlignCorners);
    writeToBuffer<DataType>(d, mType);
    assert(d == a + getSerializationSize());
}

void GridsampleIPluginV2DynamicExt::destroy() IS_NOEXCEPT
{
    DEBUG_LOG();
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void GridsampleIPluginV2DynamicExt::setPluginNamespace(const char* pluginNamespace) IS_NOEXCEPT
{
    DEBUG_LOG();
    mNamespace = pluginNamespace;
}

const char* GridsampleIPluginV2DynamicExt::getPluginNamespace() const IS_NOEXCEPT
{
    DEBUG_LOG();
    return mNamespace.c_str();
}

// -------------------- IPluginV2Ext --------------------

DataType GridsampleIPluginV2DynamicExt::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs)  const IS_NOEXCEPT
{
    DEBUG_LOG();
    // one outputs
    assert(index == 0);
    assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
    return inputTypes[0];
}

// -------------------- IPluginV2DynamicExt ------------------

IPluginV2DynamicExt* GridsampleIPluginV2DynamicExt::clone() const IS_NOEXCEPT
{
    DEBUG_LOG();
    auto plugin
        = new GridsampleIPluginV2DynamicExt(mLayerName, mInputChannel, mInputHeight, mInputWidth, 
        mGridHeight, mGridWidth, mInterpolationMode, mPaddingMode, mAlignCorners, mType);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

// To implement the output dimension, please refer to
// getOutputDimensions: https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v2_dynamic_ext.html#a2ad948f8c05a6e0ae4ab4aa92ceef311
// IExprBuilder: https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_expr_builder.html
DimsExprs GridsampleIPluginV2DynamicExt::getOutputDimensions(int32_t outputIndex, DimsExprs const *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) IS_NOEXCEPT
{
    DEBUG_LOG();

    // Validate input arguments
    assert(inputs[0].nbDims == 4);
    assert(inputs[1].nbDims == 4);
    
    // return N, C, H_g, W_g
    DimsExprs output(inputs[0]);
    output.d[2] = inputs[1].d[1];
    output.d[3] = inputs[1].d[2];
    return output;
}

bool GridsampleIPluginV2DynamicExt::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    DEBUG_LOG();

    assert(nbInputs == 2 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
    
    bool condition = inOut[pos].format == TensorFormat::kLINEAR;

    condition &= inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF;
    condition &= inOut[pos].type == inOut[0].type;
    return condition;
}

// bool GridsampleIPluginV2DynamicExt::supportsFormatCombination(int32_t pos, PluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs) IS_NOEXCEPT
// {
//     DEBUG_LOG();
//     bool is_supported = false;
//     if(pos == 0)
//     {
//         is_supported =
//             (inOut[pos].type == DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR);
//     }
//     if(pos == 1)
//     {
//         is_supported =
//             (inOut[0].type == DataType::kHALF && inOut[0].format == TensorFormat::kLINEAR &&
//             inOut[pos].type == DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR);
//     }

//     return is_supported;
// }

void GridsampleIPluginV2DynamicExt::configurePlugin(DynamicPluginTensorDesc const *inputs, int32_t nbInputs, DynamicPluginTensorDesc const *outputs, int32_t nbOutputs) IS_NOEXCEPT
{
    DEBUG_LOG();
    // This function is called by the builder prior to initialize().
    // It provides an opportunity for the layer to make algorithm choices on the basis of I/O PluginTensorDesc
    assert(nbInputs == 2);
    assert(nbOutputs == 1);

    // we only support 2d grid sampler now.
    assert(inputs[0].desc.dims.nbDims == 4);
    assert(inputs[1].desc.dims.nbDims == 4);

    mBatch = inputs[0].desc.dims.d[0];
    mInputChannel = inputs[0].desc.dims.d[1];
    mInputHeight = inputs[0].desc.dims.d[2];
    mInputWidth = inputs[0].desc.dims.d[3];
    mGridHeight = inputs[1].desc.dims.d[1];
    mGridWidth = inputs[1].desc.dims.d[2];
    mType = inputs[0].desc.type;

    assert(static_cast<int32_t>(mBatch) == inputs[1].desc.dims.d[0]);
    assert(inputs[1].desc.dims.d[3] == 2); // only supports coor = 2

}

size_t GridsampleIPluginV2DynamicExt::getWorkspaceSize(PluginTensorDesc const *inputs, int32_t nbInputs, PluginTensorDesc const *outputs, int32_t nbOutputs) const IS_NOEXCEPT
{
    DEBUG_LOG();
    // Find the workspace size required by the layer.
    // This function is called during engine startup, after initialize().
    // The workspace size returned should be sufficient for any batch size up to the maximum.
    return 0;
}

// TODO: implement by user
// The actual plugin execution func.
int32_t GridsampleIPluginV2DynamicExt::enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc, void const *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) IS_NOEXCEPT
{
    DEBUG_LOG();

    int status = -1;

    GridSamplerDataType dataType = (mType == DataType::kFLOAT ? GridSamplerDataType::GFLOAT : GridSamplerDataType::GHALF);

    std::cout << "mBatch: " << mBatch << std::endl;
    std::cout << "mInputChannel: " << mInputChannel << std::endl;
    std::cout << "mInputHeight: " << mInputHeight << std::endl;
    std::cout << "mInputWidth: " << mInputWidth << std::endl;
    std::cout << "mGridHeight: " << mGridHeight << std::endl;
    std::cout << "mGridWidth: " << mGridWidth << std::endl;
    std::cout << "mInterpolationMode: " << to_string(mInterpolationMode) << std::endl;
    std::cout << "mPaddingMode: " << to_string(mPaddingMode) << std::endl;
    std::cout << "mAlignCorners: " << mAlignCorners << std::endl;
    std::cout << "dataType: " << to_string(dataType) << std::endl;


    status = grid_sampler_2d_cuda(mBatch, inputs[0], inputs[1], outputs[0],
        mInputChannel, mInputHeight, mInputWidth, mGridHeight, mGridWidth,
        mInputChannel*mInputHeight*mInputWidth, mInputHeight*mInputWidth, mInputWidth, 1,
        mGridHeight*mGridWidth*2, mGridWidth*2, 2, 1,
        mInputChannel*mGridHeight*mGridWidth, mGridHeight*mGridWidth, mGridWidth, 1,
        mInterpolationMode, mPaddingMode, mAlignCorners, dataType, stream);

    return status;
}

// -------------------- IPluginCreator ------------------

GridsampleIPluginV2DynamicExtCreator::GridsampleIPluginV2DynamicExtCreator()
{
    DEBUG_LOG();
    // mPluginAttributes.clear();
    // // Describe GridsampleIPluginV2DynamicExt's required PluginField arguments
    // mPluginAttributes.emplace_back(PluginField("align_corners", nullptr, PluginFieldType::kINT32, 1));
    // mPluginAttributes.emplace_back(PluginField("mode", nullptr, PluginFieldType::kCHAR, 1));
    // mPluginAttributes.emplace_back(PluginField("padding_mode", nullptr, PluginFieldType::kCHAR, 1));
    
    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GridsampleIPluginV2DynamicExtCreator::getPluginName() const IS_NOEXCEPT
{
    DEBUG_LOG();
    return PLUGIN_NAME;
}

const char* GridsampleIPluginV2DynamicExtCreator::getPluginVersion() const IS_NOEXCEPT
{
    DEBUG_LOG();
    return PLUGIN_VERSION;
}

const PluginFieldCollection* GridsampleIPluginV2DynamicExtCreator::getFieldNames() IS_NOEXCEPT
{
    DEBUG_LOG();
    return &mFC;
}

IPluginV2DynamicExt* GridsampleIPluginV2DynamicExtCreator::createPlugin(const char* name, const PluginFieldCollection* fc) IS_NOEXCEPT
{
    DEBUG_LOG();

    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    int interpolationMode = 0, paddingMode = 0, alignCorners = 0;

    for (int i = 0; i < nbFields; ++i)
    {
        assert(fields[i].type == PluginFieldType::kINT32);

        if (!strcmp(fields[i].name, "interpolation_mode"))
        {
            interpolationMode = *(reinterpret_cast<const int*>(fields[i].data));
        }

        if (!strcmp(fields[i].name, "padding_mode"))
        {
            paddingMode = *(reinterpret_cast<const int*>(fields[i].data));
        }

        if (!strcmp(fields[i].name, "align_corners"))
        {
            alignCorners = *(reinterpret_cast<const int*>(fields[i].data));
        }
    }

    auto plugin = new GridsampleIPluginV2DynamicExt(name, static_cast<GridSamplerInterpolation>(interpolationMode)
        , static_cast<GridSamplerPadding>(paddingMode), static_cast<bool>(alignCorners));
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2DynamicExt* GridsampleIPluginV2DynamicExtCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) IS_NOEXCEPT
{
    DEBUG_LOG();
    return new GridsampleIPluginV2DynamicExt(name, serialData, serialLength);
}

void GridsampleIPluginV2DynamicExtCreator::setPluginNamespace(const char* libNamespace) IS_NOEXCEPT
{
    DEBUG_LOG();
    mNamespace = libNamespace;
}

const char* GridsampleIPluginV2DynamicExtCreator::getPluginNamespace() const IS_NOEXCEPT
{
    DEBUG_LOG();
    return mNamespace.c_str();
}