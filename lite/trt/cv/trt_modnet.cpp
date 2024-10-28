//
// Created by wangzijian on 10/28/24.
//

#include "trt_modnet.h"
using trtcv::TRTMODNet;

void TRTMODNet::preprocess(cv::Mat &input_mat) {
    cv::Mat ori_input_mat = input_mat;
    cv::resize(input_mat,input_mat,cv::Size(512,512));
    cv::cvtColor(input_mat,input_mat,cv::COLOR_BGR2RGB);
    if (input_mat.type() != CV_32FC3) input_mat.convertTo(input_mat, CV_32FC3);
    else input_mat = input_mat;
    input_mat = (input_mat -mean_val) * scale_val;

}



void TRTMODNet::detect(const cv::Mat &mat, types::MattingContent &content, bool remove_noise, bool minimum_post_process) {
    if (mat.empty()) return;
    cv::Mat preprocessed_mat = mat;
    preprocess(preprocessed_mat);

    const int batch_size = 1;
    const int channels = 3;
    const int input_h = preprocessed_mat.rows;
    const int input_w = preprocessed_mat.cols;
    const size_t input_size = batch_size * channels * input_h * input_w * sizeof(float);
    const size_t output_size = batch_size * channels * input_h  * input_w * sizeof(float);

    for (auto& buffer : buffers) {
        if (buffer) {
            cudaFree(buffer);
            buffer = nullptr;
        }
    }
    cudaMalloc(&buffers[0], input_size);
    cudaMalloc(&buffers[1], output_size);
    if (!buffers[0] || !buffers[1]) {
        std::cerr << "Failed to allocate CUDA memory" << std::endl;
        return;
    }

    input_node_dims = {batch_size, channels, input_h, input_w};

    std::vector<float> input;
    trtcv::utils::transform::create_tensor(preprocessed_mat,input,input_node_dims,trtcv::utils::transform::CHW);

    //3.infer
    cudaMemcpyAsync(buffers[0], input.data(), input_size,
                    cudaMemcpyHostToDevice, stream);

    nvinfer1::Dims MODNetDims;
    MODNetDims.nbDims = 4;
    MODNetDims.d[0] = batch_size;
    MODNetDims.d[1] = channels;
    MODNetDims.d[2] = input_h;
    MODNetDims.d[3] = input_w;

    auto input_tensor_name = trt_engine->getIOTensorName(0);
    auto output_tensor_name = trt_engine->getIOTensorName(1);
    trt_context->setTensorAddress(input_tensor_name, buffers[0]);
    trt_context->setTensorAddress(output_tensor_name, buffers[1]);
    trt_context->setInputShape(input_tensor_name, MODNetDims);

    bool status = trt_context->enqueueV3(stream);
    if (!status){
        std::cerr << "Failed to infer by TensorRT." << std::endl;
        return;
    }



    std::vector<float> output(batch_size * channels * input_h * input_w);
    cudaMemcpyAsync(output.data(), buffers[1], output_size,
                    cudaMemcpyDeviceToHost, stream);

    // post
    generate_matting(output.data(),mat,content, remove_noise, minimum_post_process);
}

void TRTMODNet::generate_matting(float *trt_outputs, const cv::Mat &mat, types::MattingContent &content,
                                  bool remove_noise, bool minimum_post_process) {

    const unsigned int h = mat.rows;
    const unsigned int w = mat.cols;


    const unsigned int out_h = 512;
    const unsigned int out_w = 512;

    cv::Mat alpha_pred(out_h, out_w, CV_32FC1, trt_outputs);
    cv::imwrite("/home/lite.ai.toolkit/modnet.jpg",alpha_pred);
    // post process
    if (remove_noise) trtcv::utils::remove_small_connected_area(alpha_pred,0.05f);
    // resize alpha
    if (out_h != h || out_w != w)
        // already allocated a new continuous memory after resize.
        cv::resize(alpha_pred, alpha_pred, cv::Size(w, h));
        // need clone to allocate a new continuous memory if not performed resize.
        // The memory elements point to will release after return.
    else alpha_pred = alpha_pred.clone();

    cv::Mat pmat = alpha_pred; // ref
    content.pha_mat = pmat; // auto handle the memory inside ocv with smart ref.

    if (!minimum_post_process)
    {
        // MODNet only predict Alpha, no fgr. So,
        // the fake fgr and merge mat may not need,
        // let the fgr mat and merge mat empty to
        // Speed up the post processes.
        cv::Mat mat_copy;
        mat.convertTo(mat_copy, CV_32FC3);
        // merge mat and fgr mat may not need
        std::vector<cv::Mat> mat_channels;
        cv::split(mat_copy, mat_channels);
        cv::Mat bmat = mat_channels.at(0);
        cv::Mat gmat = mat_channels.at(1);
        cv::Mat rmat = mat_channels.at(2); // ref only, zero-copy.
        bmat = bmat.mul(pmat);
        gmat = gmat.mul(pmat);
        rmat = rmat.mul(pmat);
        cv::Mat rest = 1.f - pmat;
        cv::Mat mbmat = bmat.mul(pmat) + rest * 153.f;
        cv::Mat mgmat = gmat.mul(pmat) + rest * 255.f;
        cv::Mat mrmat = rmat.mul(pmat) + rest * 120.f;
        std::vector<cv::Mat> fgr_channel_mats, merge_channel_mats;
        fgr_channel_mats.push_back(bmat);
        fgr_channel_mats.push_back(gmat);
        fgr_channel_mats.push_back(rmat);
        merge_channel_mats.push_back(mbmat);
        merge_channel_mats.push_back(mgmat);
        merge_channel_mats.push_back(mrmat);

        cv::merge(fgr_channel_mats, content.fgr_mat);
        cv::merge(merge_channel_mats, content.merge_mat);

        content.fgr_mat.convertTo(content.fgr_mat, CV_8UC3);
        content.merge_mat.convertTo(content.merge_mat, CV_8UC3);
    }

    content.flag = true;

}