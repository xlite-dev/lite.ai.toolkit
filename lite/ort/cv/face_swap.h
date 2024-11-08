//
// Created by wangzijian on 11/5/24.
//

#ifndef LITE_AI_TOOLKIT_FACE_SWAP_H
#define LITE_AI_TOOLKIT_FACE_SWAP_H
#include "lite/ort/core/ort_core.h"
#include "lite/ort/core/ort_types.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/ort/cv/face_restoration.h"

namespace ortcv{
    class LITE_EXPORTS Face_Swap : public BasicOrtHandler
    {
    public:
        explicit  Face_Swap(const std::string &_onnx_path, unsigned int _num_threads = 1):
                BasicOrtHandler(_onnx_path, _num_threads = 1){};
        ~Face_Swap() override = default;
    private:
        void preprocess(cv::Mat &target_face,std::vector<float> source_image_embeding,std::vector<cv::Point2f> target_landmark_5,std::vector<float> &processed_source_embeding,cv::Mat &preprocessed_mat);

        Ort::Value transform(const cv::Mat &mat_rs) override;

        std::pair<cv::Mat, cv::Mat> warp_face_by_face_landmark_5(cv::Mat input_mat, std::vector<cv::Point2f> face_landmark_5);

        void postprocess(cv::Mat &input_mat,std::string &output_path);

        cv::Mat create_static_box_mask(std::vector<float> crop_size);

    private:
        std::vector<cv::Point2f> face_template = {
                cv::Point2f(0.36167656, 0.40387734),
                cv::Point2f(0.63696719, 0.40235469),
                cv::Point2f(0.50019687, 0.56044219),
                cv::Point2f(0.38710391, 0.72160547),
                cv::Point2f(0.61507734, 0.72034453)
        };

        float face_mask_blur = 0.3;
        std::vector<int> face_mask_padding = {0,0,0,0};
        std::vector<cv::Mat> crop_list;
        cv::Mat affine_martix;

    public:
        void detect(cv::Mat &target_image,std::vector<float> source_face_embeding,std::vector<cv::Point2f> target_landmark_5, cv::Mat &face_swap_image);

    };
}

#endif //LITE_AI_TOOLKIT_FACE_SWAP_H
