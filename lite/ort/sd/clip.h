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
    class LITE_EXPORTS Clip : public BasicOrtHandler
    {
    public:
        explicit Clip(const std::string &_onnx_path, unsigned int _num_threads = 1) :
                BasicOrtHandler(_onnx_path, _num_threads)
        {};

        ~Clip() override = default;

    public:
        void encode_text(std::vector<std::string> input_text,std::vector<int>& output);

        void inference(std::vector<int> input,std::vector<float> &output);

        void inference(std::vector<std::string> input,std::vector<float> &output);

        Ort::Value transform(const cv::Mat &mat);
    };
}



#endif //LITE_AI_TOOLKIT_CLIP_H
