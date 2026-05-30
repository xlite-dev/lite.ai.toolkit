#ifndef FACE_RESTORATION_PREPROCESS_MANAGER_H
#define FACE_RESTORATION_PREPROCESS_MANAGER_H

#include "face_restoration_preprocess.cuh"
#include <opencv2/opencv.hpp>

// Fuses bgr2rgb + normalize + HWC->CHW into one kernel and writes the normalized RGB CHW
// tensor straight into the device inference input buffer (no D2H, no separate H2D of the
// float tensor). Reuses device + pinned staging buffers across calls.
class FaceRestorePreprocessGPU {
public:
    FaceRestorePreprocessGPU() = default;
    ~FaceRestorePreprocessGPU();

    FaceRestorePreprocessGPU(const FaceRestorePreprocessGPU&) = delete;
    FaceRestorePreprocessGPU& operator=(const FaceRestorePreprocessGPU&) = delete;

    // crop_bgr_u8: CV_8UC3 (e.g. 512x512). d_out: device float CHW buffer (the inference input).
    void run(const cv::Mat& crop_bgr_u8, float* d_out, cudaStream_t stream = nullptr);

private:
    void ensure_capacity(size_t bytes);

    unsigned char* d_crop_ = nullptr;
    unsigned char* h_pinned_ = nullptr;
    size_t cap_ = 0;
};

#endif // FACE_RESTORATION_PREPROCESS_MANAGER_H
