//
// Created by wangzijian on 11/7/24.
//

#ifndef LITE_AI_TOOLKIT_FACE_RESTORATION_H
#define LITE_AI_TOOLKIT_FACE_RESTORATION_H
#include "lite/ort/core/ort_core.h"
#include "lite/ort/core/ort_types.h"
#include "lite/ort/core/ort_utils.h"

namespace ortcv{
    class LITE_EXPORTS Face_Restoration : public BasicOrtHandler{
    public:
        explicit Face_Restoration(const std::string &_onnx_path, unsigned int _num_threads = 1):
                BasicOrtHandler(_onnx_path,_num_threads){};
        ~Face_Restoration() override = default;

    private:

        std::pair<cv::Mat, cv::Mat> warp_face_by_face_landmark_5(cv::Mat input_mat, std::vector<cv::Point2f> face_landmark_5);

        cv::Mat create_static_box_mask(std::vector<float> crop_size);

        Ort::Value transform(const cv::Mat &mat_rs) override;

        cv::Mat paste_back(const cv::Mat& temp_vision_frame,
                           const cv::Mat& crop_vision_frame,
                           const cv::Mat& crop_mask,
                           const cv::Mat& affine_matrix);


    private:
        float face_mask_blur = 0.3;
        std::vector<int> face_mask_padding = {0,0,0,0};

        std::vector<cv::Point2f> face_template = {
                cv::Point2f(0.37691676, 0.46864664),
                cv::Point2f(0.62285697, 0.46912813),
                cv::Point2f(0.50123859, 0.61331904),
                cv::Point2f(0.39308822, 0.72541100),
                cv::Point2f(0.61150205, 0.72490465)
        };

        int FACE_ENHANCER_BLEND = 80;
    public:
        void detect(cv::Mat &face_swap_image,std::vector<cv::Point2f > &target_landmarks_5 ,const std::string &face_enchaner_path);

    };
}

#endif //LITE_AI_TOOLKIT_FACE_RESTORATION_H
