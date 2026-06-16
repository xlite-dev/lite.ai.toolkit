#include "yoloface_preprocess_manager.h"
#include <cuda_runtime.h>
#include <cstring>

YoloFacePreprocessGPU::~YoloFacePreprocessGPU() {
    if (d_img_)    cudaFree(d_img_);
    if (h_pinned_) cudaFreeHost(h_pinned_);
}

void YoloFacePreprocessGPU::ensure_capacity(size_t bytes) {
    if (bytes > cap_) {
        if (d_img_)    cudaFree(d_img_);
        if (h_pinned_) cudaFreeHost(h_pinned_);
        cudaMalloc(&d_img_, bytes);
        cudaMallocHost(&h_pinned_, bytes);
        cap_ = bytes;
    }
}

void YoloFacePreprocessGPU::run(const cv::Mat& letterboxed_bgr_u8, float* d_out, cudaStream_t stream) {
    cv::Mat c = letterboxed_bgr_u8;
    if (c.type() != CV_8UC3) c.convertTo(c, CV_8UC3);
    if (!c.isContinuous()) c = c.clone();

    const int H = c.rows, W = c.cols;
    const size_t bytes = static_cast<size_t>(H) * W * 3;
    ensure_capacity(bytes);

    std::memcpy(h_pinned_, c.data, bytes);
    cudaMemcpyAsync(d_img_, h_pinned_, bytes, cudaMemcpyHostToDevice, stream);

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    yoloface_preprocess_kernel<<<grid, block, 0, stream>>>(d_img_, d_out, H, W);

    cudaStreamSynchronize(stream);
}
