//
// Created by wangzijian on 11/4/24.
//

#include "face_recognizer.h"
using ortcv::Face_Recognizer;


cv::Mat Face_Recognizer::preprocess(cv::Mat &input_mat, std::vector<cv::Point2f> &face_landmark_5,cv::Mat &preprocessed_mat) {
    cv::Mat crop_image;
    cv::Mat affine_martix;

    std::tie(crop_image,affine_martix) = face_utils::warp_face_by_face_landmark_5(input_mat,face_landmark_5,face_utils::ARCFACE_112_V2);
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