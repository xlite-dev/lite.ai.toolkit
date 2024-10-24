//
// Created by wangzijian on 10/24/24.
//

#ifndef LITE_AI_TOOLKIT_REAL_ESR_GAN_H
#define LITE_AI_TOOLKIT_REAL_ESR_GAN_H
#include "lite/ort/core/ort_core.h"
#include "vector"
#include "opencv2/opencv.hpp"

namespace ortcv{
    class LITE_EXPORTS RealESRGAN : public BasicOrtHandler{
    public:
        explicit RealESRGAN(const std::string &_onnx_path, unsigned int _num_threads = 1):
                BasicOrtHandler(_onnx_path,_num_threads){};
        ~RealESRGAN() override = default;
    private:
        void preprocess(const cv::Mat& frame,cv::Mat &output_mat);
        Ort::Value transform(const cv::Mat &mat_rs) override;
        void postprocess(std::vector<Ort::Value> &ort_outputs,const std::string &output_path);
    public:
        void detect(const cv::Mat &input_mat,const std::string &output_path);

    };
}



#endif //LITE_AI_TOOLKIT_REAL_ESR_GAN_H
