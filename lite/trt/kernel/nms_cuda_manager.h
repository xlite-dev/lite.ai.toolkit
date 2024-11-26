// nms_cuda_manager.hpp
#pragma once
#include <vector>
#include <memory>
#include <stdexcept>
#include "lite/types.h"

class NMSCudaManager {
public:
    NMSCudaManager();
    ~NMSCudaManager();

    // 禁用拷贝构造和赋值运算符，防止意外的资源复制
    NMSCudaManager(const NMSCudaManager&) = delete;
    NMSCudaManager& operator=(const NMSCudaManager&) = delete;

    // 初始化CUDA资源，支持动态调整
    void init(size_t max_boxes = 1024);

    // 安全的NMS执行方法
    std::vector<int> perform_nms(
            const std::vector<lite::types::Boxf>& boxes,
            const std::vector<float>& confidences,
            float nms_thresh
    );

private:
    // 资源释放方法
    void release_resources();

    // CUDA内存指针
    float* d_boxes = nullptr;      // 设备内存：框
    int* d_result = nullptr;       // 设备内存：结果
    int* h_result = nullptr;       // 主机内存：结果

    size_t max_boxes_num = 0;      // 最大框数
    bool is_initialized = false;   // 初始化标志
};