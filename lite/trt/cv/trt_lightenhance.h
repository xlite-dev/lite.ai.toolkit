//
// Created by wangzijian on 10/22/24.
//

#ifndef LITE_AI_TOOLKIT_TRT_LIGHTENHANCE_H
#define LITE_AI_TOOLKIT_TRT_LIGHTENHANCE_H
#include "lite/trt/core/trt_core.h"
#include "lite/trt/core/trt_utils.h"


namespace trtcv{
    class LITE_EXPORTS TRTLightEnhance : public BasicTRTHandler{
    public:
        explicit TRTLightEnhance(const std::string& _trt_model_path,unsigned int _num_threads = 1):
                BasicTRTHandler(_trt_model_path, _num_threads)
        {};
    private:
        int ori_input_width;
        int ori_input_height;
        void preprocess(const cv::Mat& frame,cv::Mat &output_mat);
        void postprocess(float *trt_outputs,const std::string &output_path);
    public:
        void detect(const cv::Mat &input_mat,const std::string &output_path);
    };
}



#endif //LITE_AI_TOOLKIT_TRT_LIGHTENHANCE_H
