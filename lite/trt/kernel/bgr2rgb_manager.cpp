//
// Created by root on 12/3/24.
//

#include "bgr2rgb_manager.h"

void launch_bgr2rgb(const cv::Mat& bgr_image, cv::Mat& rgb_image) {
    int width = bgr_image.cols;
    int height = bgr_image.rows;

    // 确保输出图像已经分配好内存并且大小正确
    if (rgb_image.empty()) {
        rgb_image.create(height, width, CV_8UC3);
    }

    // 分配设备内存
    uchar3* d_bgr;
    uchar3* d_rgb;
    cudaMalloc(&d_bgr, width * height * sizeof(uchar3));
    cudaMalloc(&d_rgb, width * height * sizeof(uchar3));

    // 复制输入图像到设备
    cudaMemcpy(d_bgr, bgr_image.data, width * height * sizeof(uchar3),
               cudaMemcpyHostToDevice);

    // 设置kernel启动参数
    dim3 block(16, 16);  // 可以根据需要调整
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    // 启动kernel
    bgr2rgb_kernel<<<grid, block>>>(d_bgr, d_rgb, width, height);

    // 复制结果回主机
    cudaMemcpy(rgb_image.data, d_rgb, width * height * sizeof(uchar3),
               cudaMemcpyDeviceToHost);

    // 清理设备内存
    cudaFree(d_bgr);
    cudaFree(d_rgb);
}





