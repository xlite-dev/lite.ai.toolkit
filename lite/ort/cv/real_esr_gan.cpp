//
// Created by wangzijian on 10/24/24.
//

#include "real_esr_gan.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/utils.h"
using ortcv::RealESRGAN;

void RealESRGAN::preprocess(const cv::Mat &frame, cv::Mat &output_mat) {
    cv::cvtColor(frame,output_mat,cv::COLOR_BGR2RGB);
    output_mat.convertTo(output_mat,CV_32FC3,1 / 255.f);
}

Ort::Value RealESRGAN::transform(const cv::Mat &mat_rs) {
    input_node_dims[0] = 1;
    input_node_dims[1] = mat_rs.channels();
    input_node_dims[2] = mat_rs.rows;
    input_node_dims[3] = mat_rs.cols;

    return ortcv::utils::transform::create_tensor(
            mat_rs, input_node_dims, memory_info_handler,
            input_values_handler, ortcv::utils::transform::CHW);
}

void RealESRGAN::detect(const cv::Mat &input_mat, const std::string &output_path) {
    if (input_mat.empty()) return;
    int ori_input_width = input_mat.cols;
    int ori_input_height = input_mat.rows;
    cv::Mat preprocessed;
    preprocess(input_mat,preprocessed);
    Ort::Value input_tensor = transform(preprocessed);
    Ort::RunOptions runOptions;

    // 2.infer
    auto output_tensors = ort_session->Run(
            runOptions, input_node_names.data(),
            &input_tensor, 1, output_node_names.data(), num_outputs
    );

    // 3.postprocess
    postprocess(output_tensors,output_path);
}

void RealESRGAN::postprocess(std::vector<Ort::Value> &ort_outputs, const std::string &output_path) {
    float *pdata = ort_outputs[0].GetTensorMutableData<float>();
    std::vector<int64_t> out_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const int out_h = out_shape[2];
    const int out_w = out_shape[3];
    const int channel_step = out_h * out_w;
    cv::Mat bmat(out_h, out_w, CV_32FC1, pdata);
    cv::Mat gmat(out_h, out_w, CV_32FC1, pdata + channel_step);
    cv::Mat rmat(out_h, out_w, CV_32FC1, pdata + 2 * channel_step);
    bmat *= 255.f;
    gmat *= 255.f;
    rmat *= 255.f;
    std::vector<cv::Mat> channel_mats = {rmat, gmat, bmat};
    cv::Mat dstimg;
    cv::merge(channel_mats,dstimg);
    dstimg.convertTo(dstimg, CV_8UC3);
    cv::imwrite(output_path,dstimg);
}