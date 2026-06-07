#ifndef PASTE_BACK_MANAGER_H
#define PASTE_BACK_MANAGER_H

#include "paste_back.cuh"
#include "device_frame.h"
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

    // blend_alpha folds an optional face-enhancer blend into the mask: 1.0 = plain paste,
    // 0.8 = restoration's blend_frame(target 0.2 / paste 0.8) collapsed into one kernel.
    cv::Mat paste_back(const cv::Mat& temp_vision_frame,   // CV_8UC3 (auto-converted otherwise)
                       const cv::Mat& crop_vision_frame,   // CV_32FC3, 0..255
                       const cv::Mat& crop_mask,           // CV_32FC1, 0..1
                       const cv::Mat& affine_matrix,       // 2x3, original->crop
                       cudaStream_t stream = nullptr,
                       float blend_alpha = 1.0f);

    // Device-resident temp: the full frame is ALREADY on the device (e.g. a DeviceFrame), so the
    // temp H2D is skipped — the kernel reads d_temp directly. crop/mask/affine still come from host.
    cv::Mat paste_back(const unsigned char* d_temp, int W, int H,   // device BGR uint8 full frame
                       const cv::Mat& crop_vision_frame,
                       const cv::Mat& crop_mask,
                       const cv::Mat& affine_matrix,
                       cudaStream_t stream = nullptr,
                       float blend_alpha = 1.0f);

    // Device-OUTPUT: paste straight into a DeviceFrame (no D2H) so the result stays GPU-resident
    // for the next stage. Host-temp variant (uploads temp) and device-temp variant.
    void paste_back_to_device(const cv::Mat& temp_vision_frame,
                              const cv::Mat& crop_vision_frame, const cv::Mat& crop_mask,
                              const cv::Mat& affine_matrix, DeviceFrame& out,
                              cudaStream_t stream = nullptr, float blend_alpha = 1.0f);
    void paste_back_to_device(const unsigned char* d_temp, int W, int H,
                              const cv::Mat& crop_vision_frame, const cv::Mat& crop_mask,
                              const cv::Mat& affine_matrix, DeviceFrame& out,
                              cudaStream_t stream = nullptr, float blend_alpha = 1.0f);

private:
    void ensure_capacity(size_t temp_bytes, size_t crop_bytes,
                         size_t mask_bytes, size_t out_bytes);
    // Uploads temp into d_temp_ (grows it as needed) and returns the device pointer.
    const unsigned char* upload_temp(const cv::Mat& temp_vision_frame, cudaStream_t stream);
    // Core: crop/mask/affine H2D + kernel into d_out (no D2H). d_temp is the full-frame input
    // (d_temp_ after an H2D, or a caller device pointer); d_out is the kernel's output buffer.
    void run_core(const unsigned char* d_temp, int W, int H,
                  const cv::Mat& crop_vision_frame, const cv::Mat& crop_mask,
                  const cv::Mat& affine_matrix, cudaStream_t stream, float blend_alpha,
                  unsigned char* d_out);
    // run_core into d_out_ then D2H -> a fresh host Mat.
    cv::Mat run(const unsigned char* d_temp, int W, int H,
                const cv::Mat& crop_vision_frame, const cv::Mat& crop_mask,
                const cv::Mat& affine_matrix, cudaStream_t stream, float blend_alpha);

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
