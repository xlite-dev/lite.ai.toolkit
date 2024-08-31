//
// Created by root on 8/28/24.
//

#ifndef LITE_AI_TOOLKIT_TRT_VAE_H
#define LITE_AI_TOOLKIT_TRT_VAE_H
#include "lite/trt/core/trt_config.h"
#include "lite/trt/core/trt_logger.h"
#include "lite/trt/core/trt_utils.h"
#include "cuda_fp16.h"

namespace trtsd{
    class TRTVae {
    public:
        TRTVae(const std::string &_engine_path);
        ~TRTVae();

    private:
        std::unique_ptr<nvinfer1::IRuntime> trt_runtime;
        std::unique_ptr<nvinfer1::ICudaEngine> trt_engine;
        std::unique_ptr<nvinfer1::IExecutionContext> trt_context;

        Logger trt_logger;
        std::vector<void*> buffers;
        cudaStream_t stream;

    private:
        std::vector<int64_t> input_node_dims = {1,4,64,64};
        int output_size =  1 * 3 * 512 * 512;
        std::vector<int64_t> output_node_dims = {1,3,512,512};
        const char * input_names = "latent";
        const char * output_names = "images";

        const char* trt_model_path = nullptr;

    private:
        void trt_save_vector_as_image(const std::vector<float>& output_vector, int height, int width, const std::string& filename);
    public:
        void inference();

        void inference(const std::vector<float> &unet_input,const std::string save_path);



    };
}



#endif //LITE_AI_TOOLKIT_TRT_VAE_H
