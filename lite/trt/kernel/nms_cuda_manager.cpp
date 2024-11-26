// nms_cuda_manager.cpp
#include "nms_cuda_manager.h"
#include "nms_kernel.cuh"
#include <cuda_runtime.h>

// 宏定义：检查CUDA操作是否成功
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
        } \
    } while(0)

NMSCudaManager::NMSCudaManager() {}

NMSCudaManager::~NMSCudaManager() {
    release_resources();
}

void NMSCudaManager::init(size_t max_boxes) {
    // 如果已经初始化且新的大小不超过当前大小，则直接返回
    if (is_initialized && max_boxes <= max_boxes_num) {
        return;
    }

    // 先释放现有资源
    release_resources();

    try {
        // 分配设备内存
        CUDA_CHECK(cudaMalloc(&d_boxes, max_boxes * 5 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_result, max_boxes * sizeof(int)));

        // 分配主机内存
        h_result = new int[max_boxes];

        max_boxes_num = max_boxes;
        is_initialized = true;
    }
    catch (const std::exception& e) {
        // 初始化失败时，确保资源被正确释放
        release_resources();
        throw;
    }
}

void NMSCudaManager::release_resources() {
    if (d_boxes) {
        CUDA_CHECK(cudaFree(d_boxes));
        d_boxes = nullptr;
    }

    if (d_result) {
        CUDA_CHECK(cudaFree(d_result));
        d_result = nullptr;
    }

    if (h_result) {
        delete[] h_result;
        h_result = nullptr;
    }

    max_boxes_num = 0;
    is_initialized = false;
}



std::vector<int> NMSCudaManager::perform_nms(
        const std::vector<lite::types::Boxf>& boxes,
        const std::vector<float>& confidences,
        float nms_thresh
) {
    // 安全性检查
    if (boxes.size() != confidences.size()) {
        throw std::invalid_argument("Box and confidence sizes must match");
    }

    // 初始化或调整资源大小
    const int num_boxes = boxes.size();
    if (true ) {
        init(fmax(num_boxes, max_boxes_num * 2));
    }

    // 准备数据
    std::vector<float> box_data(num_boxes * 5);
    for (int i = 0; i < num_boxes; ++i) {
        box_data[i * 5] = boxes[i].x1;
        box_data[i * 5 + 1] = boxes[i].y1;
        box_data[i * 5 + 2] = boxes[i].x2;
        box_data[i * 5 + 3] = boxes[i].y2;
        box_data[i * 5 + 4] = confidences[i];
    }

    // 拷贝数据到GPU
    CUDA_CHECK(cudaMemcpy(d_boxes, box_data.data(), num_boxes * 5 * sizeof(float), cudaMemcpyHostToDevice));

    // 设置CUDA kernel参数
    const int block_size = 256;
    const int grid_size = (num_boxes + block_size - 1) / block_size;

    // 启动kernel
    nms_kernel<<<grid_size, block_size>>>(d_boxes, num_boxes, nms_thresh, d_result);
    CUDA_CHECK(cudaGetLastError());

    // 等待kernel执行完成
    CUDA_CHECK(cudaDeviceSynchronize());

    // 拷贝结果回CPU
    CUDA_CHECK(cudaMemcpy(h_result, d_result, num_boxes * sizeof(int), cudaMemcpyDeviceToHost));

    // 收集保留的索引
    std::vector<int> keep_indices;
    for (int i = 0; i < num_boxes; ++i) {
        if (h_result[i] == 1) {
            keep_indices.push_back(i);
        }
    }

    return keep_indices;
}