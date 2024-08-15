//
// Created by wangzijian on 8/5/24.
//

#ifndef LITE_AI_TOOLKIT_CLIP_H
#define LITE_AI_TOOLKIT_CLIP_H

#include "lite/ort/core/ort_core.h"
#include "vocab.h"
#include "iostream"
#include "vector"
#include "tokenizer.h"

namespace ortsd
{
    class  Clip
    {
    public:
        Clip(const std::string &_onnx_path, unsigned int _num_threads = 1);

        ~Clip();

    private:
        Ort::Env ort_env;
        Ort::Session *ort_session = nullptr;
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);

        unsigned int num_inputs = 1;
        const LITEORT_CHAR *onnx_path = nullptr;
        const char *log_id = nullptr;
        bool context_is_update = false;
        const unsigned int num_threads; // initialize at runtime.
        const int input_axes = 77;
        const int output_tensor_size = 77 * 768;
        const int end_flag_num = 49407;

        std::vector<const char *> input_node_names = {
                "input_ids"
        };

        std::vector<const char *> output_node_names = {
                "text_embeddings"
        };

    public:
        void encode_text(std::vector<std::string> input_text, std::vector<std::vector<int>> &output);

        void inference(std::vector<std::string> input,std::vector<std::vector<float>> &output);

    };
}



#endif //LITE_AI_TOOLKIT_CLIP_H
