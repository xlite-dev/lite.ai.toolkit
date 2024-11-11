//
// Created by wangzijian on 11/4/24.
//

#include "face_recognizer.h"
using ortcv::Face_Recognizer;

std::pair<cv::Mat, cv::Mat> Face_Recognizer::warp_face_by_face_landmark_5(cv::Mat input_mat, std::vector<cv::Point2f> face_landmark_5)
{

    std::vector<cv::Point2f> normed_template;
    for(auto current_template : face_utils::face_template_112)
    {
        current_template.x = current_template.x * 112;
        current_template.y = current_template.y * 112;
        normed_template.emplace_back(current_template);
    }

    cv::Mat inliers;
    cv::Mat affine_matrix = cv::estimateAffinePartial2D(
            face_landmark_5,
            normed_template,
            inliers,
            cv::RANSAC,
            100
    );


    if (affine_matrix.empty()) {
        throw std::runtime_error("Failed to estimate affine transformation");
    }


    cv::Mat crop_img;
    cv::warpAffine(
            input_mat,
            crop_img,
            affine_matrix,
            cv::Size(112, 112),
            cv::INTER_AREA,
            cv::BORDER_REPLICATE
    );

    return std::make_pair(crop_img, affine_matrix);
}


cv::Mat Face_Recognizer::preprocess(cv::Mat &input_mat, std::vector<cv::Point2f> &face_landmark_5,cv::Mat &preprocessed_mat) {
    cv::Mat crop_image;
    cv::Mat affine_martix;

    std::tie(crop_image,affine_martix) = warp_face_by_face_landmark_5(input_mat,face_landmark_5);
    crop_image.convertTo(crop_image,CV_32FC3, 1.0f / 127.5f,-1.0);
    cv::cvtColor(crop_image,crop_image,cv::COLOR_BGR2RGB);

    return crop_image;

}


Ort::Value Face_Recognizer::transform(const cv::Mat &mat_rs) {
    input_node_dims[0] = 1;
    input_node_dims[1] = mat_rs.channels();
    input_node_dims[2] = mat_rs.rows;
    input_node_dims[3] = mat_rs.cols;

    return ortcv::utils::transform::create_tensor(
            mat_rs, input_node_dims, memory_info_handler,
            input_values_handler, ortcv::utils::transform::CHW);
}

void Face_Recognizer::detect(cv::Mat &input_mat, std::vector<cv::Point2f> &face_landmark_5) {
    cv::Mat ori_image = input_mat.clone();

    cv::Mat crop_image = preprocess(input_mat,face_landmark_5,ori_image);
    Ort::Value input_tensor = transform(crop_image);
    Ort::RunOptions runOptions;

    // 2.infer
    auto output_tensors = ort_session->Run(
            runOptions, input_node_names.data(),
            &input_tensor, 1, output_node_names.data(), num_outputs
    );

    float *pdata = output_tensors[0].GetTensorMutableData<float>();
    std::vector<int64_t> out_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    std::vector<float> output(pdata, pdata + 512);

    float norm = 0.0f;
    for (const auto &val : output) {
        norm += val * val;
    }
    norm = std::sqrt(norm);

    for (auto &val : output) {
        val /= norm;
    }

    std::cout<<"done!"<<std::endl;

}

void Face_Recognizer::detect(cv::Mat &input_mat, std::vector<cv::Point2f> &face_landmark_5, std::vector<float> &embeding) {
    cv::Mat ori_image = input_mat.clone();

    cv::Mat crop_image = preprocess(input_mat,face_landmark_5,ori_image);
    Ort::Value input_tensor = transform(crop_image);
    Ort::RunOptions runOptions;

    // 2.infer
    auto output_tensors = ort_session->Run(
            runOptions, input_node_names.data(),
            &input_tensor, 1, output_node_names.data(), num_outputs
    );

    float *pdata = output_tensors[0].GetTensorMutableData<float>();
    std::vector<int64_t> out_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    embeding.assign(pdata,pdata + 512);
    std::vector<float> normal_embeding(pdata,pdata + 512);


    float norm = 0.0f;
    for (const auto &val : normal_embeding) {
        norm += val * val;
    }
    norm = std::sqrt(norm);

    for (auto &val : normal_embeding) {
        val /= norm;
    }

    std::cout<<"done!"<<std::endl;
}