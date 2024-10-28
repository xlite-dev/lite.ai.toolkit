//
// Created by wangzijian on 10/28/24.
//

#ifndef LITE_AI_TOOLKIT_TRT_MODNET_H
#define LITE_AI_TOOLKIT_TRT_MODNET_H

#include "lite/trt/core/trt_core.h"
#include "lite/trt/core/trt_utils.h"

namespace trtcv{
    class LITE_EXPORTS TRTMODNet : public BasicTRTHandler{
    public:
        explicit TRTMODNet(const std::string& _trt_model_path,unsigned int _num_threads = 1):
                BasicTRTHandler(_trt_model_path, _num_threads)
        {};
    private:
        static constexpr const float mean_val = 127.5f; // RGB
        static constexpr const float scale_val = 1.f / 127.5f;
    private:
        void preprocess(cv::Mat &input_mat);

        void generate_matting(float *trt_outputs,
                              const cv::Mat &mat, types::MattingContent &content,
                              bool remove_noise = false, bool minimum_post_process = false);
    public:
        void detect(const cv::Mat &mat, types::MattingContent &content, bool remove_noise = false,
                    bool minimum_post_process = false);
    };
}



#endif //LITE_AI_TOOLKIT_TRT_MODNET_H
