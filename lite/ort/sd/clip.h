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

//namespace ortsd
//{
//    class LITE_EXPORTS Clip : public BasicOrtHandler
//    {
//    public:
//        explicit Clip(const std::string &_onnx_path, unsigned int _num_threads = 1) :
//                BasicOrtHandler(_onnx_path, _num_threads)
//        {};
//
//        ~Clip() override = default;
//
//    public:
//        void encode_text(std::vector<std::string> input_text,std::vector<int>& output);
//
//        void inference(std::vector<int> input,std::vector<float> &output);
//
//        void inference(std::vector<std::string> input,std::vector<float> &output);
//
//        Ort::Value transform(const cv::Mat &mat);
//    };
//}

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

        std::vector<const char *> input_node_names = {
                "TEXT"
        };

        std::vector<const char *> output_node_names = {
                "TEXT_EMBEDDING"
        };

    public:
        void encode_text(std::vector<std::string> input_text, std::vector<std::vector<int>> &output);

        void inference(std::vector<int> input,std::vector<float> &output);

        void inference(std::vector<std::string> input,std::vector<std::vector<float>> &output);

        Ort::Value transform(const cv::Mat &mat);
    };
}



#endif //LITE_AI_TOOLKIT_CLIP_H
