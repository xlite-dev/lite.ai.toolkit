//
// Created by wangzijian on 11/14/24.
//

#ifndef LITE_AI_TOOLKIT_TRT_FACE_RESTORATION_H
#define LITE_AI_TOOLKIT_TRT_FACE_RESTORATION_H
#include "lite/trt/core/trt_core.h"
#include "lite/trt/core/trt_utils.h"
#include "lite/trt/core/trt_config.h"
#include "lite/ort/cv/face_utils.h"
#include "lite/trt/kernel/face_restoration_postprocess_manager.h"
namespace trtcv{
    class LITE_EXPORTS TRTFaceFusionFaceRestoration : BasicTRTHandler{
    public:
        explicit TRTFaceFusionFaceRestoration(const std::string& _trt_model_path,unsigned int _num_threads = 1) :
                BasicTRTHandler(_trt_model_path,_num_threads){};;
    public:
        void detect(cv::Mat &face_swap_image,std::vector<cv::Point2f > &target_landmarks_5 ,const std::string &face_enchaner_path);

    };
}

#endif //LITE_AI_TOOLKIT_TRT_FACE_RESTORATION_H
