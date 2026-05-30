#ifndef PASTE_BACK_MANAGER_H
#define PASTE_BACK_MANAGER_H

#include "paste_back.cuh"
#include <opencv2/opencv.hpp>

// Old CPU-heavy version (two full-frame warpAffine + per-call malloc/sync copies); kept for A/B.
cv::Mat launch_paste_back(const cv::Mat& temp_vision_frame,
                          const cv::Mat& crop_vision_frame,
                          const cv::Mat& crop_mask,
                          const cv::Mat& affine_matrix);

// GPU fused version: inverse-mapping sampling + blend entirely in the kernel, reused device
// buffers, pinned + async copies. Numerically equivalent to launch_paste_back; returns full-frame BGR uint8.
class PasteBackGPU {
public:
    PasteBackGPU() = default;
    ~PasteBackGPU();

    PasteBackGPU(const PasteBackGPU&) = delete;
    PasteBackGPU& operator=(const PasteBackGPU&) = delete;

    cv::Mat paste_back(const cv::Mat& temp_vision_frame,   // CV_8UC3 (auto-converted otherwise)
                       const cv::Mat& crop_vision_frame,   // CV_32FC3, 0..255
                       const cv::Mat& crop_mask,           // CV_32FC1, 0..1
                       const cv::Mat& affine_matrix,       // 2x3, original->crop
                       cudaStream_t stream = nullptr);

private:
    void ensure_capacity(size_t temp_bytes, size_t crop_bytes,
                         size_t mask_bytes, size_t out_bytes);

    unsigned char* d_temp_ = nullptr;
    unsigned char* d_out_  = nullptr;
    float* d_crop_   = nullptr;
    float* d_mask_   = nullptr;
    float* d_affine_ = nullptr;            // 6 floats
    unsigned char* h_temp_pinned_ = nullptr;
    unsigned char* h_out_pinned_  = nullptr;
    size_t cap_temp_ = 0, cap_crop_ = 0, cap_mask_ = 0, cap_out_ = 0;
};

#endif // PASTE_BACK_MANAGER_H
