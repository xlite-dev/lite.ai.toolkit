//
// Created by wangzijian on 10/22/24.
//

#ifndef LITE_AI_TOOLKIT_LIGHT_ENHANCE_H
#define LITE_AI_TOOLKIT_LIGHT_ENHANCE_H
#include "lite/ort/core/ort_core.h"
#include "opencv2/opencv.hpp"
#include "vector"
namespace ortcv{
    class LITE_EXPORTS LightEnhance : public BasicOrtHandler{
    public:
        explicit LightEnhance(const std::string &_onnx_path, unsigned int _num_threads = 1):
                BasicOrtHandler(_onnx_path,_num_threads){};
        ~LightEnhance() override = default;
    private:
        int ori_input_width;
        int ori_input_height;
        void preprocess(const cv::Mat& frame,cv::Mat &output_mat);
        Ort::Value transform(const cv::Mat &mat_rs) override;
        void postprocess(std::vector<Ort::Value> &ort_outputs,const std::string &output_path);
    public:
        void detect(const cv::Mat &input_mat,const std::string &output_path);
    };
}


#endif //LITE_AI_TOOLKIT_LIGHT_ENHANCE_H
