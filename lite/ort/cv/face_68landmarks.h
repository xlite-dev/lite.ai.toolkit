//
// Created by wangzijian on 11/1/24.
//

#ifndef LITE_AI_TOOLKIT_FACE_68LANDMARKS_H
#define LITE_AI_TOOLKIT_FACE_68LANDMARKS_H
#include "lite/ort/core/ort_core.h"
#include "lite/ort/core/ort_types.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/utils.h"
#include "lite/ort/cv/face_utils.h"
#include "algorithm"

namespace ortcv{
    class LITE_EXPORTS Face_68Landmarks : public BasicOrtHandler{
    public:
        explicit  Face_68Landmarks(const std::string &_onnx_path, unsigned int _num_threads = 1):
                BasicOrtHandler(_onnx_path, _num_threads = 1){};

        ~Face_68Landmarks() override = default;

    private:
        cv::Mat affine_matrix;
        cv::Mat img_with_landmarks;

    private:
        void preprocess(const lite::types::Boxf &bouding_box,const cv::Mat &input_mat,cv::Mat &crop_img);

        Ort::Value transform(const cv::Mat &mat_rs) override;

        void postprocess(std::vector<Ort::Value> &ort_outputs, std::vector<cv::Point2f> &face_landmark_5of68);


    public:

        void detect(const cv::Mat &input_mat,const lite::types::BoundingBoxType<float, float> &bbox, std::vector<cv::Point2f> &face_landmark_5of68);



    };


}

#endif //LITE_AI_TOOLKIT_FACE_68LANDMARKS_H
