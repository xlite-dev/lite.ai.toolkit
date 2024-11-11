//
// Created by wangzijian on 11/11/24.
//

#ifndef LITE_AI_TOOLKIT_FACE_UTILS_H
#define LITE_AI_TOOLKIT_FACE_UTILS_H
#include "opencv2/opencv.hpp"
#include <fstream>
#pragma once

namespace face_utils
{


    cv::Mat paste_back(const cv::Mat& temp_vision_frame,
                       const cv::Mat& crop_vision_frame,
                       const cv::Mat& crop_mask,
                       const cv::Mat& affine_matrix);

    std::pair<cv::Mat, cv::Mat> warp_face_by_translation(const cv::Mat& temp_img,cv::Point2f& translation,
                                                         float scale, const cv::Size& crop_size);

    std::vector<float> dot_product(const std::vector<float>& vec,
                                   const std::vector<float>& matrix,
                                   int matrix_cols);

    std::pair<cv::Mat, cv::Mat> warp_face_by_face_landmark_5(cv::Mat input_mat,
                                                             std::vector<cv::Point2f> face_landmark_5,unsigned int type);

    std::vector<cv::Point2f> convert_face_landmark_68_to_5(const std::vector<cv::Point2f>& landmark_68);

    cv::Mat blend_frame(const cv::Mat &target_image, const cv::Mat &paste_frame);

    cv::Mat create_static_box_mask(std::vector<float> crop_size);

    void normalize(std::vector<float>& vec);

    float calculate_norm(const std::vector<float>& vec);

    std::vector<float> load_npy(const std::string& filename);

    // 需要把下面三个vector整合在一起

    extern const std::vector<std::vector<cv::Point2f>> face_template_vector;

    extern const std::vector<cv::Point2f> face_template_128;

    extern const std::vector<cv::Point2f> face_template_112;

    extern const std::vector<cv::Point2f> face_template_512;

    enum FaceType {
        ARCFACE_112_V2 = 112,
        ARCFACE_128_V2 = 128,
        FFHQ_512 = 512
    };

}

#endif //LITE_AI_TOOLKIT_FACE_UTILS_H
