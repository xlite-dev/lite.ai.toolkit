#ifndef LITE_AI_TOOLKIT_WARP_AFFINE_NPP_H
#define LITE_AI_TOOLKIT_WARP_AFFINE_NPP_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// GPU affine warp via NPP (nppiWarpAffine), reusing device + pinned staging buffers.
// Replaces cv::warpAffine for the face-crop warps. Same affine convention as
// cv::warpAffine(src, dst, M): M is the 2x3 source->template transform from
// estimateAffinePartial2D. (Device-resident variant returning a device pointer comes next;
// this host->host version is the de-risk step to validate the NPP convention + quality.)
class WarpAffineNpp {
public:
    WarpAffineNpp() = default;
    ~WarpAffineNpp();
    WarpAffineNpp(const WarpAffineNpp&) = delete;
    WarpAffineNpp& operator=(const WarpAffineNpp&) = delete;

    // frame_bgr_u8: CV_8UC3 full frame. affine_2x3: 2x3 (CV_32F/64F). out_size: crop is out_size x out_size.
    cv::Mat warp(const cv::Mat& frame_bgr_u8, const cv::Mat& affine_2x3, int out_size);

    // Device-resident variant: warps into the internal device buffer and returns a device pointer
    // to the out_size x out_size interleaved BGR uint8 crop (no D2H). The frame H2D + NPP warp run
    // on `stream`; caller must use the same stream for the consumer (no sync here). The returned
    // pointer is owned by this object and valid until the next warp_to_device/warp call.
    const unsigned char* warp_to_device(const cv::Mat& frame_bgr_u8, const cv::Mat& affine_2x3,
                                        int out_size, cudaStream_t stream = nullptr);

private:
    void ensure(size_t src_bytes, size_t dst_bytes);
    unsigned char* d_src_ = nullptr;
    unsigned char* d_dst_ = nullptr;
    unsigned char* h_src_pinned_ = nullptr;
    unsigned char* h_dst_pinned_ = nullptr;
    size_t cap_src_ = 0, cap_dst_ = 0;
};

#endif // LITE_AI_TOOLKIT_WARP_AFFINE_NPP_H
