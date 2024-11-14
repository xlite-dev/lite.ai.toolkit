//
// Created by wangzijian on 11/13/24.
//

#include "trt_face_recognizer.h"
using trtcv::TRTFaceFusionFaceRecognizer;

cv::Mat TRTFaceFusionFaceRecognizer::preprocess(cv::Mat &input_mat, std::vector<cv::Point2f> &face_landmark_5,
                                                cv::Mat &preprocessed_mat) {
    cv::Mat crop_image;
    cv::Mat affine_martix;

    std::tie(crop_image,affine_martix) = face_utils::warp_face_by_face_landmark_5(input_mat,face_landmark_5,face_utils::ARCFACE_112_V2);
    crop_image.convertTo(crop_image,CV_32FC3, 1.0f / 127.5f,-1.0);
    cv::cvtColor(crop_image,crop_image,cv::COLOR_BGR2RGB);

    return crop_image;
}


void TRTFaceFusionFaceRecognizer::detect(cv::Mat &input_mat, std::vector<cv::Point2f> &face_landmark_5,
                                         std::vector<float> &embeding) {
    cv::Mat ori_image = input_mat.clone();

    cv::Mat crop_image = preprocess(input_mat,face_landmark_5,ori_image);


    std::vector<float> input_vector;

    trtcv::utils::transform::create_tensor(crop_image,input_vector,input_node_dims,
                                           trtcv::utils::transform::CHW);

    cudaMemcpyAsync(buffers[0], input_vector.data(), input_node_dims[0] * input_node_dims[1] * input_node_dims[2] * input_node_dims[3] * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // 在推理之前同步流，確保數據完全拷貝
    cudaStreamSynchronize(stream);
    bool status = trt_context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    if (!status){
        std::cerr << "Failed to infer by TensorRT." << std::endl;
        return;
    }

    std::vector<float> output(output_node_dims[0][0] * output_node_dims[0][1]);
    cudaMemcpyAsync(output.data(), buffers[1], output_node_dims[0][0] * output_node_dims[0][1] * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    embeding.assign(output.begin(),output.end());
    std::vector<float> normal_embeding(output.begin(),output.end());


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