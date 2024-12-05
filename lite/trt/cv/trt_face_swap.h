//
// Created by wangzijian on 11/13/24.
//

#ifndef LITE_AI_TOOLKIT_TRT_FACE_SWAP_H
#define LITE_AI_TOOLKIT_TRT_FACE_SWAP_H
#include "lite/ort/cv/face_utils.h"
#include "lite/trt/core/trt_core.h"
#include "lite/trt/core/trt_utils.h"
#include "lite/trt/core/trt_types.h"
#include "lite/trt/kernel/face_swap_postproces_manager.h"
#include "lite/trt/kernel/paste_back_manager.h"

namespace trtcv{
    class LITE_EXPORTS TRTFaceFusionFaceSwap : BasicTRTHandler{
    public:
        explicit TRTFaceFusionFaceSwap(const std::string& _trt_model_path,unsigned int _num_threads = 1):
                BasicTRTHandler(_trt_model_path,_num_threads){};
    private:
        void preprocess(cv::Mat &target_face,std::vector<float> source_image_embeding,std::vector<cv::Point2f> target_landmark_5,
                        std::vector<float> &processed_source_embeding,cv::Mat &preprocessed_mat);

    private:
        std::vector<cv::Mat> crop_list;
        cv::Mat affine_martix;
    public:
        void detect(cv::Mat &target_image,std::vector<float> source_face_embeding,std::vector<cv::Point2f> target_landmark_5,
                    cv::Mat &face_swap_image);

    };
}


#endif //LITE_AI_TOOLKIT_TRT_FACE_SWAP_H
