//
// Created by wangzijian on 8/7/24.
//

#ifndef LITE_AI_TOOLKIT_TRT_CLIP_H
#define LITE_AI_TOOLKIT_TRT_CLIP_H
#include "lite/trt/core/trt_config.h"
#include "lite/trt/core/trt_logger.h"

namespace trtsd
{
    class TRTClip{
    public:
        TRTClip(const std::string &_onnx_path);
        ~TRTClip();

    private:
        std::unique_ptr<nvinfer1::IRuntime> trt_runtime;
        std::unique_ptr<nvinfer1::ICudaEngine> trt_engine;
        std::unique_ptr<nvinfer1::IExecutionContext> trt_context;

        Logger trt_logger;
        std::vector<void*> buffers;
        cudaStream_t stream;

        std::vector<int64_t> input_node_dims;
        std::vector<std::vector<int64_t>> output_node_dims;
        const int output_tensor_size = 77 * 768;
        const int input_axes = 77;
        const char * input_names = "input_ids";
        const char * output_names = "text_embeddings";
        const int end_flag_num = 49407;

        const char* trt_model_path = nullptr;
        const char* log_id = nullptr;
    public:

        void encode_text(std::vector<std::string> input_text, std::vector<std::vector<int>> &output);

        void inference(std::vector<std::string> input,std::vector<std::vector<float>> &output);

    };


}



#endif //LITE_AI_TOOLKIT_TRT_CLIP_H
