#ifndef YOLOFACE_PREPROCESS_MANAGER_H
#define YOLOFACE_PREPROCESS_MANAGER_H

#include "yoloface_preprocess.cuh"
#include <opencv2/opencv.hpp>

// Fuses normalize (v/128 - 127.5/128) + BGR HWC->CHW into one kernel and writes the tensor
// straight into the device inference input buffer — replacing the CPU split / 3x convertTo /
// merge / create_tensor + the separate float H2D. Reuses device + pinned staging buffers.
class YoloFacePreprocessGPU {
public:
    YoloFacePreprocessGPU() = default;
    ~YoloFacePreprocessGPU();

    YoloFacePreprocessGPU(const YoloFacePreprocessGPU&) = delete;
    YoloFacePreprocessGPU& operator=(const YoloFacePreprocessGPU&) = delete;

    // letterboxed_bgr_u8: CV_8UC3, already resized + padded to the network input size.
    // d_out: device float CHW buffer (the inference input).
    void run(const cv::Mat& letterboxed_bgr_u8, float* d_out, cudaStream_t stream = nullptr);

private:
    void ensure_capacity(size_t bytes);

    unsigned char* d_img_    = nullptr;
    unsigned char* h_pinned_ = nullptr;
    size_t cap_ = 0;
};

#endif // YOLOFACE_PREPROCESS_MANAGER_H
