#include "face_restoration_preprocess_manager.h"
#include <cuda_runtime.h>
#include <cstring>

FaceRestorePreprocessGPU::~FaceRestorePreprocessGPU() {
    if (d_crop_)  cudaFree(d_crop_);
    if (h_pinned_) cudaFreeHost(h_pinned_);
}

void FaceRestorePreprocessGPU::ensure_capacity(size_t bytes) {
    if (bytes > cap_) {
        if (d_crop_)  cudaFree(d_crop_);
        if (h_pinned_) cudaFreeHost(h_pinned_);
        cudaMalloc(&d_crop_, bytes);
        cudaMallocHost(&h_pinned_, bytes);
        cap_ = bytes;
    }
}

void FaceRestorePreprocessGPU::run(const cv::Mat& crop_bgr_u8, float* d_out, cudaStream_t stream,
                                   float scale, float bias) {
    cv::Mat c = crop_bgr_u8;
    if (c.type() != CV_8UC3) c.convertTo(c, CV_8UC3);
    if (!c.isContinuous()) c = c.clone();

    const int H = c.rows, W = c.cols;
    const size_t bytes = static_cast<size_t>(H) * W * 3;
    ensure_capacity(bytes);

    std::memcpy(h_pinned_, c.data, bytes);
    cudaMemcpyAsync(d_crop_, h_pinned_, bytes, cudaMemcpyHostToDevice, stream);

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    face_restoration_preprocess_kernel<<<grid, block, 0, stream>>>(d_crop_, d_out, H, W, scale, bias);

    cudaStreamSynchronize(stream);
}

void FaceRestorePreprocessGPU::run_device(const unsigned char* d_crop, int H, int W,
                                          float* d_out, cudaStream_t stream,
                                          float scale, float bias) {
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    face_restoration_preprocess_kernel<<<grid, block, 0, stream>>>(d_crop, d_out, H, W, scale, bias);
    cudaStreamSynchronize(stream);
}
