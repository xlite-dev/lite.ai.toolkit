#ifndef LITE_AI_TOOLKIT_DEVICE_FRAME_H
#define LITE_AI_TOOLKIT_DEVICE_FRAME_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// A full frame kept resident in device memory (HWC, BGR, uint8) for the device pipeline:
// upload once at entry, hand the raw device pointer between GPU stages (NPP warp, paste-back)
// without bouncing through host, download once at exit. Reuses its device + pinned staging
// buffers across frames (no per-frame cudaMalloc).
class DeviceFrame {
public:
    DeviceFrame() = default;
    ~DeviceFrame();
    DeviceFrame(const DeviceFrame&) = delete;
    DeviceFrame& operator=(const DeviceFrame&) = delete;

    // Host BGR uint8 -> device (1 H2D, via pinned staging for true async). Does not sync;
    // the consuming GPU op must run on the same stream.
    void upload(const cv::Mat& bgr_u8, cudaStream_t stream = nullptr);
    // Device -> host BGR uint8 (1 D2H, syncs). Returns a fresh CV_8UC3 Mat.
    cv::Mat download(cudaStream_t stream = nullptr) const;

    // Size the device buffer for a w*h*3 BGR uint8 frame WITHOUT uploading (e.g. as a kernel
    // output target). Returns the device pointer.
    unsigned char* prepare(int w, int h) { ensure(w, h); return d_; }

    unsigned char* data() { return d_; }
    const unsigned char* data() const { return d_; }
    int width()  const { return w_; }
    int height() const { return h_; }
    bool empty() const { return d_ == nullptr; }

private:
    void ensure(int w, int h);
    unsigned char* d_ = nullptr;
    mutable unsigned char* h_pinned_ = nullptr;
    size_t cap_ = 0;
    int w_ = 0, h_ = 0;
};

#endif // LITE_AI_TOOLKIT_DEVICE_FRAME_H
