//
// Created by wangzijian on 10/25/24.
//

#include "trt_realesrgan.h"
using trtcv::TRTRealESRGAN;

void TRTRealESRGAN::preprocess(const cv::Mat &frame, cv::Mat &output_mat) {
    cv::cvtColor(frame,output_mat,cv::COLOR_BGR2RGB);
    output_mat.convertTo(output_mat,CV_32FC3,1 / 255.f);
}


void TRTRealESRGAN::detect(const cv::Mat &input_mat, const std::string &output_path) {
    if (input_mat.empty()) return;

    ori_input_width = input_mat.cols;
    ori_input_height = input_mat.rows;

    cv::Mat preprocessed_mat;
    preprocess(input_mat, preprocessed_mat);

    const int batch_size = 1;
    const int channels = 3;
    const int input_h = preprocessed_mat.rows;
    const int input_w = preprocessed_mat.cols;
    const size_t input_size = batch_size * channels * input_h * input_w * sizeof(float);
    const size_t output_size = batch_size * channels * input_h * 4 * input_w * 4 * sizeof(float);

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
    trtcv::utils::transform::create_tensor(preprocessed_mat, input, input_node_dims, trtcv::utils::transform::CHW);

    cudaError_t status = cudaMemcpyAsync(buffers[0], input.data(), input_size,
                                         cudaMemcpyHostToDevice, stream);
    if (status != cudaSuccess) {
        std::cerr << "Input copy failed: " << cudaGetErrorString(status) << std::endl;
        return;
    }
    cudaStreamSynchronize(stream);

    nvinfer1::Dims ESRGANDims;
    ESRGANDims.nbDims = 4;
    ESRGANDims.d[0] = batch_size;
    ESRGANDims.d[1] = channels;
    ESRGANDims.d[2] = input_h;
    ESRGANDims.d[3] = input_w;

    auto input_tensor_name = trt_engine->getIOTensorName(0);
    auto output_tensor_name = trt_engine->getIOTensorName(1);
    trt_context->setTensorAddress(input_tensor_name, buffers[0]);
    trt_context->setTensorAddress(output_tensor_name, buffers[1]);

    trt_context->setInputShape(input_tensor_name, ESRGANDims);

    bool infer_status = trt_context->enqueueV3(stream);
    if (!infer_status) {
        std::cerr << "TensorRT inference failed!" << std::endl;
        return;
    }
    cudaStreamSynchronize(stream);

    const size_t total_output_elements = batch_size * channels * input_h * 4 * input_w * 4;
    std::vector<float> output(total_output_elements);

    status = cudaMemcpyAsync(output.data(), buffers[1], output_size,
                             cudaMemcpyDeviceToHost, stream);
    if (status != cudaSuccess) {
        std::cerr << "Output copy failed: " << cudaGetErrorString(status) << std::endl;
        return;
    }
    cudaStreamSynchronize(stream);

    postprocess(output.data(), output_path);
}

void TRTRealESRGAN::postprocess(float *trt_outputs, const std::string &output_path) {
    const int out_h = ori_input_height * 4;
    const int out_w = ori_input_width * 4;
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
    cv::imwrite(output_path,dstimg);
}


