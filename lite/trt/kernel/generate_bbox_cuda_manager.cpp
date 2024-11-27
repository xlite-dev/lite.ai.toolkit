//
// Created by wangzijian on 11/26/24.
//

#include "generate_bbox_cuda_manager.h"
// Kernel launch wrapper function
void launch_yolov8_postprocess(
        float* trt_outputs,
        int number_of_boxes,
        float conf_threshold,
        float ratio_height,
        float ratio_width,
        lite::types::BoundingBoxType<float, float>* output_boxes,
        int max_output_boxes
) {
    // 计算grid和block尺寸
    int block_size = 256;
    int grid_size = (number_of_boxes + block_size - 1) / block_size;

    // 分配设备内存
    lite::types::BoundingBoxType<float, float>* d_output_boxes;
    int* d_output_count;

    cudaMalloc(&d_output_boxes, max_output_boxes * sizeof(lite::types::BoundingBoxType<float, float>));
    cudaMalloc(&d_output_count, sizeof(int));
    cudaMemset(d_output_count, 0, sizeof(int));

    // 启动内核
    yolov8_postprocess_kernel<<<grid_size, block_size>>>(
            trt_outputs,
            number_of_boxes,
            conf_threshold,
            ratio_height,
            ratio_width,
            d_output_boxes,
            d_output_count
    );

    // 同步和错误检查
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    // 复制输出数据
    int h_output_count;
    cudaMemcpy(&h_output_count, d_output_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_boxes, d_output_boxes, h_output_count * sizeof(lite::types::BoundingBoxType<float, float>), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_output_boxes);
    cudaFree(d_output_count);
}