//
// Created by root on 8/28/24.
//

#ifndef LITE_AI_TOOLKIT_TRT_UNET_H
#define LITE_AI_TOOLKIT_TRT_UNET_H
#include "lite/trt/core/trt_config.h"
#include "lite/trt/core/trt_logger.h"
#include "lite/trt/core/trt_utils.h"
#include "lite/ort/sd/ddimscheduler.h"
#include "cuda_fp16.h"
#include "trt_clip.h"
#include <random>
#include <algorithm>

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
        std::vector<int64_t> input_node_dims_sample = {2, 4 , 64, 64};
        std::vector<int64_t> input_node_dims_time_step = {1};
        std::vector<int64_t> input_node_dims_encoder_hidden_states = {2, 77, 768};
        std::vector<int64_t> unet_output_dims = {2, 4, 64, 64};
        std::vector<int> noise_pred_dims = {1, 4, 64, 64};
        std::vector<const char*> input_names = {"sample", "timestep", "encoder_hidden_states"};
        const char * output_names = "latent";
        const char* trt_model_path = nullptr;

    private:
        int unet_outputsize = 2 * 4 * 64 * 64;
        int final_latent_outputsize = 1 * 4 * 64 * 64;
        int noise_pred_outputsize = 1 * 4 * 64 * 64;
        float cfg_scale =  7.5f;

    private:
        std::vector<float> convertToFloat(const std::vector<half>& half_vec);


    public:
        void inference();

        void inference(const std::vector<std::vector<float>> &clip_output,std::vector<float> &latent,std::string scheduler_config_path);




    };
}



#endif //LITE_AI_TOOLKIT_TRT_UNET_H
