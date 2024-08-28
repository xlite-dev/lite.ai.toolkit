//
// Created by root on 8/28/24.
//

#ifndef LITE_AI_TOOLKIT_TRT_UNET_H
#define LITE_AI_TOOLKIT_TRT_UNET_H
#include "lite/trt/core/trt_config.h"
#include "lite/trt/core/trt_logger.h"
#include "cuda_fp16.h"
#include "trt_clip.h"
#include <random>

namespace trtsd{
    class TRTUNet{
    public:
        TRTUNet(const std::string &_engine_path);
        ~TRTUNet();
    private:
        std::unique_ptr<nvinfer1::IRuntime> trt_runtime;
        std::unique_ptr<nvinfer1::ICudaEngine> trt_engine;
        std::unique_ptr<nvinfer1::IExecutionContext> trt_context;

        Logger trt_logger;
        std::vector<void*> buffers;
        cudaStream_t stream;
        std::vector<int64_t> input_node_dims;
        std::vector<int64_t> output_node_dims;

        std::vector<const char*> input_names = {"sample", "timestep", "encoder_hidden_states"};

        const char * output_names = "latent";

        const char* trt_model_path = nullptr;

    public:
        void inference();

    };
}



#endif //LITE_AI_TOOLKIT_TRT_UNET_H
