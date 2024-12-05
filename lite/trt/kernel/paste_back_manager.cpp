#include "paste_back_manager.h"
#include <cuda_runtime.h>

cv::Mat launch_paste_back(const cv::Mat& temp_vision_frame,
                          const cv::Mat& crop_vision_frame,
                          const cv::Mat& crop_mask,
                          const cv::Mat& affine_matrix) {
    // 转换为float类型
    cv::Mat temp_float, crop_float, mask_float;
    temp_vision_frame.convertTo(temp_float, CV_32F);
    crop_vision_frame.convertTo(crop_float, CV_32F);
    crop_mask.convertTo(mask_float, CV_32F);

    // 获取仿射变换的逆矩阵
    cv::Mat inverse_matrix;
    cv::invertAffineTransform(affine_matrix, inverse_matrix);

    // 获取目标尺寸
    cv::Size temp_size(temp_vision_frame.cols, temp_vision_frame.rows);

    // 对mask和crop_frame进行反向仿射变换
    cv::Mat inverse_mask, inverse_vision_frame;
    cv::warpAffine(mask_float, inverse_mask, inverse_matrix, temp_size);
    cv::warpAffine(crop_float, inverse_vision_frame, inverse_matrix, temp_size);

    // 阈值处理
    cv::threshold(inverse_mask, inverse_mask, 1.0, 1.0, cv::THRESH_TRUNC);
    cv::threshold(inverse_mask, inverse_mask, 0.0, 0.0, cv::THRESH_TOZERO);

    // 准备CUDA内存
    int width = temp_vision_frame.cols;
    int height = temp_vision_frame.rows;
    int channels = temp_vision_frame.channels();
    size_t total_size = width * height * channels * sizeof(float);
    size_t mask_size = width * height * sizeof(float);

    float *d_inverse_vision_frame, *d_temp_frame, *d_inverse_mask, *d_output;
    cudaMalloc(&d_inverse_vision_frame, total_size);
    cudaMalloc(&d_temp_frame, total_size);
    cudaMalloc(&d_inverse_mask, mask_size);
    cudaMalloc(&d_output, total_size);

    // 复制数据到GPU
    cudaMemcpy(d_inverse_vision_frame, inverse_vision_frame.ptr<float>(), total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp_frame, temp_float.ptr<float>(), total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inverse_mask, inverse_mask.ptr<float>(), mask_size, cudaMemcpyHostToDevice);

    // 设置kernel参数
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // 启动kernel
    paste_back_kernel<<<grid, block>>>(d_inverse_vision_frame,
                                       d_temp_frame,
                                       d_inverse_mask,
                                       d_output,
                                       width, height, channels);

    // 创建输出Mat
    cv::Mat result(height, width, CV_32FC3);

    // 复制结果回主机
    cudaMemcpy(result.ptr<float>(), d_output, total_size, cudaMemcpyDeviceToHost);

    // 清理GPU内存
    cudaFree(d_inverse_vision_frame);
    cudaFree(d_temp_frame);
    cudaFree(d_inverse_mask);
    cudaFree(d_output);

    // 如果需要，转换回原始类型
    cv::Mat final_result;
    if(temp_vision_frame.type() != CV_32F) {
        result.convertTo(final_result, temp_vision_frame.type());
    } else {
        final_result = result;
    }

    return final_result;
}
