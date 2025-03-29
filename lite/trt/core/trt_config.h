//
// Created by wangzijian on 24-7-11.
//

#ifndef LITE_AI_TOOLKIT_TRT_CONFIG_H
#define LITE_AI_TOOLKIT_TRT_CONFIG_H
#include "trt_defs.h"
// common header
#include "lite/lite.ai.headers.h"
// TensorRT backend
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"
#include "NvOnnxParser.h"
// define new namespace
namespace trtcore{
    // Define the pipeline modes using an enum class (preferred for strong typing)
    enum class PipelineMode {
        TXT2IMG = 0,
        IMG2IMG = 1
        // Add more modes here if needed
    };
}


#endif //LITE_AI_TOOLKIT_TRT_CONFIG_H
