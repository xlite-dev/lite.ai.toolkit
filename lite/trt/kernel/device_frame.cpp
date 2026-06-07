#include "device_frame.h"
#include <cstring>

DeviceFrame::~DeviceFrame() {
    if (d_)        cudaFree(d_);
    if (h_pinned_) cudaFreeHost(h_pinned_);
}

void DeviceFrame::ensure(int w, int h) {
    const size_t bytes = static_cast<size_t>(w) * h * 3;
    if (bytes > cap_) {
        if (d_)        cudaFree(d_);
        if (h_pinned_) cudaFreeHost(h_pinned_);
        cudaMalloc(&d_, bytes);
        cudaMallocHost(&h_pinned_, bytes);
        cap_ = bytes;
    }
    w_ = w; h_ = h;
}

void DeviceFrame::upload(const cv::Mat& bgr_u8, cudaStream_t stream) {
    cv::Mat f = bgr_u8;
    if (f.type() != CV_8UC3) f.convertTo(f, CV_8UC3);
    if (!f.isContinuous()) f = f.clone();

    ensure(f.cols, f.rows);
    const size_t bytes = static_cast<size_t>(w_) * h_ * 3;
    std::memcpy(h_pinned_, f.data, bytes);
    cudaMemcpyAsync(d_, h_pinned_, bytes, cudaMemcpyHostToDevice, stream);
}

cv::Mat DeviceFrame::download(cudaStream_t stream) const {
    cv::Mat out(h_, w_, CV_8UC3);
    const size_t bytes = static_cast<size_t>(w_) * h_ * 3;
    cudaMemcpyAsync(h_pinned_, d_, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::memcpy(out.data, h_pinned_, bytes);
    return out;
}
