//
// Created by root on 3/27/25.
//

#ifndef LITE_AI_TOOLKIT_TRT_VAE_ENCODER_H
#define LITE_AI_TOOLKIT_TRT_VAE_ENCODER_H
#include "lite/trt/core/trt_config.h"
#include "lite/trt/core/trt_logger.h"
#include "lite/trt/core/trt_utils.h"
#include "cuda_fp16.h"
namespace trtsd{
    class TRTVaeEncoder{
    public:
        TRTVaeEncoder(const std::string &_engine_path);
        ~TRTVaeEncoder();

    private:
        std::unique_ptr<nvinfer1::IRuntime> trt_runtime;
        std::unique_ptr<nvinfer1::ICudaEngine> trt_engine;
        std::unique_ptr<nvinfer1::IExecutionContext> trt_context;

        Logger trt_logger;
        std::vector<void*> buffers;
        cudaStream_t stream;

    private:
        std::vector<int64_t> input_node_dims = {1,3,512,512};
        int output_size =  1 * 4 * 64 * 64;
        std::vector<int64_t> output_node_dims = {1,3,64,64};
        const char * input_names = "images";
        const char * output_names = "latent";
        const char* trt_model_path = nullptr;
        float vae_scaling_factor = 0.1825;

    public:
        void inference(const std::vector<float> &input_images,std::vector<float> &output_latents);

    };
}

#endif //LITE_AI_TOOLKIT_TRT_VAE_ENCODER_H
