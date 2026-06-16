#include "warp_affine_npp.h"
#include <nppi_geometry_transforms.h>
#include <nppcore.h>
#include <cstring>

// Fill the NPP 2x3 coefficient array from a cv 2x3 affine (same forward src->dst convention).
static void fill_coeffs(const cv::Mat& affine_2x3, double aCoeffs[2][3]) {
    cv::Mat M64;
    affine_2x3.convertTo(M64, CV_64F);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j) aCoeffs[i][j] = M64.at<double>(i, j);
}

WarpAffineNpp::~WarpAffineNpp() {
    if (d_src_)        cudaFree(d_src_);
    if (d_dst_)        cudaFree(d_dst_);
    if (h_src_pinned_) cudaFreeHost(h_src_pinned_);
    if (h_dst_pinned_) cudaFreeHost(h_dst_pinned_);
}

void WarpAffineNpp::ensure(size_t src_bytes, size_t dst_bytes) {
    if (src_bytes > cap_src_) {
        if (d_src_)        cudaFree(d_src_);
        if (h_src_pinned_) cudaFreeHost(h_src_pinned_);
        cudaMalloc(&d_src_, src_bytes);
        cudaMallocHost(&h_src_pinned_, src_bytes);
        cap_src_ = src_bytes;
    }
    if (dst_bytes > cap_dst_) {
        if (d_dst_)        cudaFree(d_dst_);
        if (h_dst_pinned_) cudaFreeHost(h_dst_pinned_);
        cudaMalloc(&d_dst_, dst_bytes);
        cudaMallocHost(&h_dst_pinned_, dst_bytes);
        cap_dst_ = dst_bytes;
    }
}

cv::Mat WarpAffineNpp::warp(const cv::Mat& frame_bgr_u8, const cv::Mat& affine_2x3, int out_size) {
    cv::Mat src = frame_bgr_u8;
    if (src.type() != CV_8UC3) src.convertTo(src, CV_8UC3);
    if (!src.isContinuous()) src = src.clone();

    const int SW = src.cols, SH = src.rows;
    const int DW = out_size, DH = out_size;
    const size_t src_bytes = static_cast<size_t>(SW) * SH * 3;
    const size_t dst_bytes = static_cast<size_t>(DW) * DH * 3;
    ensure(src_bytes, dst_bytes);

    std::memcpy(h_src_pinned_, src.data, src_bytes);
    cudaMemcpy(d_src_, h_src_pinned_, src_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_dst_, 0, dst_bytes);

    double aCoeffs[2][3];
    fill_coeffs(affine_2x3, aCoeffs);

    NppiSize srcSize{SW, SH};
    NppiRect srcROI{0, 0, SW, SH};
    NppiRect dstROI{0, 0, DW, DH};
    // Same convention as cv::warpAffine(src,dst,M) (no WARP_INVERSE_MAP): forward src->dst M.
    nppiWarpAffine_8u_C3R(d_src_, srcSize, SW * 3, srcROI,
                          d_dst_, DW * 3, dstROI,
                          aCoeffs, NPPI_INTER_LINEAR);

    cudaMemcpy(h_dst_pinned_, d_dst_, dst_bytes, cudaMemcpyDeviceToHost);
    cv::Mat dst(DH, DW, CV_8UC3, h_dst_pinned_);
    return dst.clone();
}

const unsigned char* WarpAffineNpp::warp_to_device(const cv::Mat& frame_bgr_u8,
                                                   const cv::Mat& affine_2x3,
                                                   int out_size, cudaStream_t stream) {
    cv::Mat src = frame_bgr_u8;
    if (src.type() != CV_8UC3) src.convertTo(src, CV_8UC3);
    if (!src.isContinuous()) src = src.clone();

    const int SW = src.cols, SH = src.rows;
    const int DW = out_size, DH = out_size;
    const size_t src_bytes = static_cast<size_t>(SW) * SH * 3;
    const size_t dst_bytes = static_cast<size_t>(DW) * DH * 3;
    ensure(src_bytes, dst_bytes);

    // frame H2D on the caller's stream (still needed — frame lives on host); the warped crop then
    // stays on device (d_dst_) to feed the next GPU stage directly. No D2H, no sync here.
    std::memcpy(h_src_pinned_, src.data, src_bytes);
    cudaMemcpyAsync(d_src_, h_src_pinned_, src_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(d_dst_, 0, dst_bytes, stream);

    double aCoeffs[2][3];
    fill_coeffs(affine_2x3, aCoeffs);

    NppiSize srcSize{SW, SH};
    NppiRect srcROI{0, 0, SW, SH};
    NppiRect dstROI{0, 0, DW, DH};
    nppSetStream(stream);
    nppiWarpAffine_8u_C3R(d_src_, srcSize, SW * 3, srcROI,
                          d_dst_, DW * 3, dstROI,
                          aCoeffs, NPPI_INTER_LINEAR);
    return d_dst_;
}

const unsigned char* WarpAffineNpp::warp_device_to_device(const unsigned char* d_frame,
                                                         int SW, int SH,
                                                         const cv::Mat& affine_2x3,
                                                         int out_size, cudaStream_t stream) {
    const int DW = out_size, DH = out_size;
    const size_t dst_bytes = static_cast<size_t>(DW) * DH * 3;
    ensure(0, dst_bytes);   // src is the caller's device frame; only the dst buffer is ours

    cudaMemsetAsync(d_dst_, 0, dst_bytes, stream);

    double aCoeffs[2][3];
    fill_coeffs(affine_2x3, aCoeffs);

    NppiSize srcSize{SW, SH};
    NppiRect srcROI{0, 0, SW, SH};
    NppiRect dstROI{0, 0, DW, DH};
    nppSetStream(stream);
    nppiWarpAffine_8u_C3R(d_frame, srcSize, SW * 3, srcROI,
                          d_dst_, DW * 3, dstROI,
                          aCoeffs, NPPI_INTER_LINEAR);
    return d_dst_;
}
