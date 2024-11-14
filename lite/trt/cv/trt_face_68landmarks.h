//
// Created by wangzijian on 11/12/24.
//

#ifndef LITE_AI_TOOLKIT_TRT_FACE_68LANDMARKS_H
#define LITE_AI_TOOLKIT_TRT_FACE_68LANDMARKS_H
#include "lite/ort/cv/face_utils.h"
#include "lite/trt/core/trt_core.h"
#include "lite/trt/core/trt_utils.h"
#include "lite/trt/core/trt_types.h"
#include "algorithm"


namespace trtcv{
    class LITE_EXPORTS TRTFaceFusionFace68Landmarks : public BasicTRTHandler{
    public:
        explicit TRTFaceFusionFace68Landmarks(const std::string& _trt_model_path,unsigned int _num_threads = 1):
        BasicTRTHandler(_trt_model_path,_num_threads){};
    private:
        cv::Mat affine_matrix;
        cv::Mat img_with_landmarks;
    private:
        void preprocess(const lite::types::Boxf &bouding_box,const cv::Mat &input_mat,cv::Mat &crop_img);

        void postprocess(float *trt_outputs, std::vector<cv::Point2f> &face_landmark_5of68);

    public:

        void detect(const cv::Mat &input_mat,const lite::types::BoundingBoxType<float, float> &bbox, std::vector<cv::Point2f> &face_landmark_5of68);



    };
}


#endif //LITE_AI_TOOLKIT_TRT_FACE_68LANDMARKS_H
