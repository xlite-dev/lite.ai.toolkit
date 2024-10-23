//
// Created by wangzijian on 10/22/24.
//

#include "trt_lightenhance.h"
using trtcv::TRTLightEnhance;

void TRTLightEnhance::preprocess(const cv::Mat &input_mat,cv::Mat &output_mat) {
    cv::cvtColor(input_mat,output_mat,cv::COLOR_BGR2RGB);
    cv::resize(output_mat,output_mat,cv::Size(input_node_dims[3],input_node_dims[2]));
    output_mat.convertTo(output_mat,CV_32FC3,1 / 255.f);
}

void TRTLightEnhance::postprocess(float *trt_outputs, const std::string &output_path) {
    const int out_h = output_node_dims[0][2];
    const int out_w = output_node_dims[0][3];
    const int channel_step = out_h * out_w;
    cv::Mat bmat(out_h, out_w, CV_32FC1, trt_outputs);
    cv::Mat gmat(out_h, out_w, CV_32FC1, trt_outputs + channel_step);
    cv::Mat rmat(out_h, out_w, CV_32FC1, trt_outputs + 2 * channel_step);
    bmat *= 255.f;
    gmat *= 255.f;
    rmat *= 255.f;
    std::vector<cv::Mat> channel_mats = {rmat, gmat, bmat};
    cv::Mat dstimg;
    cv::merge(channel_mats,dstimg);
    dstimg.convertTo(dstimg, CV_8UC3);
    cv::resize(dstimg, dstimg, cv::Size(ori_input_width,ori_input_height));
    cv::imwrite(output_path,dstimg);
}

void TRTLightEnhance::detect(const cv::Mat &input_mat, const std::string &output_path) {
    if (input_mat.empty()) return;
    ori_input_height = input_mat.rows;
    ori_input_width = input_mat.cols;

    cv::Mat preprocessed_mat;
    preprocess(input_mat,preprocessed_mat);

    // 2.trans to input vector
    std::vector<float> input;
    trtcv::utils::transform::create_tensor(preprocessed_mat,input,input_node_dims,trtcv::utils::transform::CHW);

    //3.infer
    cudaMemcpyAsync(buffers[0], input.data(), input_node_dims[0] * input_node_dims[1] * input_node_dims[2] * input_node_dims[3] * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    bool status = trt_context->enqueueV3(stream);
    if (!status){
        std::cerr << "Failed to infer by TensorRT." << std::endl;
        return;
    }

    std::vector<float> output(output_node_dims[0][0] * output_node_dims[0][1] * output_node_dims[0][2]* output_node_dims[0][3]);
    cudaMemcpyAsync(output.data(), buffers[1], output_node_dims[0][0] * output_node_dims[0][1] * output_node_dims[0][2] * output_node_dims[0][3] * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

    // postprocess
    postprocess(output.data(),output_path);


}