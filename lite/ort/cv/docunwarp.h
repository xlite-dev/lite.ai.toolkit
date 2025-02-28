//
// Created by wangzijian on 2/26/25.
//

#ifndef LITE_AI_TOOLKIT_DOCUNWARP_H
#define LITE_AI_TOOLKIT_DOCUNWARP_H

#include "lite/ort/core/ort_core.h"
#include <stdexcept> // 用于异常处理
namespace ortcv{
    class LITE_EXPORTS DocUnWarp : public BasicOrtHandler{
    public:
        explicit DocUnWarp(const std::string &_onnx_path, unsigned int _num_threads = 1):
                BasicOrtHandler(_onnx_path,_num_threads)
        {};
        ~DocUnWarp() override = default;
    private:
        Ort::Value transform(const cv::Mat &mat) override;

        void preprocess(const cv::Mat &input_image,cv::Mat &preprocessed_image);

        void postprocess( std::vector<Ort::Value> &pred,cv::Mat &postprocess_mat);

    private:
        std::vector<float> pad_info;

    public:
        void detect(const cv::Mat & input_image,cv::Mat &out_image);

    };
}

#endif //LITE_AI_TOOLKIT_DOCUNWARP_H
