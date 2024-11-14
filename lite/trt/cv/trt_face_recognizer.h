//
// Created by wangzijian on 11/13/24.
//

#ifndef LITE_AI_TOOLKIT_TRT_FACE_RECOGNIZER_H
#define LITE_AI_TOOLKIT_TRT_FACE_RECOGNIZER_H
#include "lite/ort/cv/face_utils.h"
#include "lite/trt/core/trt_core.h"
#include "lite/trt/core/trt_utils.h"
#include "lite/trt/core/trt_types.h"

namespace trtcv{
    class  LITE_EXPORTS TRTFaceFusionFaceRecognizer : BasicTRTHandler{
    public:
        explicit TRTFaceFusionFaceRecognizer(const std::string& _trt_model_path,unsigned int _num_threads = 1):
                BasicTRTHandler(_trt_model_path,_num_threads){};
    private:
        cv::Mat  preprocess(cv::Mat &input_mat, std::vector<cv::Point2f> &face_landmark_5,cv::Mat &preprocessed_mat);

    public:
        void detect(cv::Mat &input_mat,std::vector<cv::Point2f> &face_landmark_5,std::vector<float> &embeding);

    };
}



#endif //LITE_AI_TOOLKIT_TRT_FACE_RECOGNIZER_H
