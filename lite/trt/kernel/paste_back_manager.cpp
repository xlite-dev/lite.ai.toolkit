#include "paste_back_manager.h"
#include <cuda_runtime.h>
#include <cstring>

cv::Mat launch_paste_back(const cv::Mat& temp_vision_frame,
                          const cv::Mat& crop_vision_frame,
                          const cv::Mat& crop_mask,
                          const cv::Mat& affine_matrix) {
    // convert to float
    cv::Mat temp_float, crop_float, mask_float;
    temp_vision_frame.convertTo(temp_float, CV_32F);
    crop_vision_frame.convertTo(crop_float, CV_32F);
    crop_mask.convertTo(mask_float, CV_32F);

    // inverse of the affine transform
    cv::Mat inverse_matrix;
    cv::invertAffineTransform(affine_matrix, inverse_matrix);

    // target (full-frame) size
    cv::Size temp_size(temp_vision_frame.cols, temp_vision_frame.rows);

    // inverse-warp the mask and crop frame back to the full frame
    cv::Mat inverse_mask, inverse_vision_frame;
    cv::warpAffine(mask_float, inverse_mask, inverse_matrix, temp_size);
    cv::warpAffine(crop_float, inverse_vision_frame, inverse_matrix, temp_size);

    // clamp mask to [0, 1]
    cv::threshold(inverse_mask, inverse_mask, 1.0, 1.0, cv::THRESH_TRUNC);
    cv::threshold(inverse_mask, inverse_mask, 0.0, 0.0, cv::THRESH_TOZERO);

    // allocate CUDA memory
    int width = temp_vision_frame.cols;
    int height = temp_vision_frame.rows;
    int channels = temp_vision_frame.channels();
    size_t total_size = width * height * channels * sizeof(float);
    size_t mask_size = width * height * sizeof(float);

    float *d_inverse_vision_frame, *d_temp_frame, *d_inverse_mask, *d_output;
    cudaMalloc(&d_inverse_vision_frame, total_size);
    cudaMalloc(&d_temp_frame, total_size);
    cudaMalloc(&d_inverse_mask, mask_size);
    cudaMalloc(&d_output, total_size);

    // copy data to GPU
    cudaMemcpy(d_inverse_vision_frame, inverse_vision_frame.ptr<float>(), total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp_frame, temp_float.ptr<float>(), total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inverse_mask, inverse_mask.ptr<float>(), mask_size, cudaMemcpyHostToDevice);

    // kernel launch config
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // launch kernel
    paste_back_kernel<<<grid, block>>>(d_inverse_vision_frame,
                                       d_temp_frame,
                                       d_inverse_mask,
                                       d_output,
                                       width, height, channels);

    // output Mat
    cv::Mat result(height, width, CV_32FC3);

    // copy result back to host
    cudaMemcpy(result.ptr<float>(), d_output, total_size, cudaMemcpyDeviceToHost);

    // free GPU memory
    cudaFree(d_inverse_vision_frame);
    cudaFree(d_temp_frame);
    cudaFree(d_inverse_mask);
    cudaFree(d_output);

    // convert back to the original type if needed
    cv::Mat final_result;
    if(temp_vision_frame.type() != CV_32F) {
        result.convertTo(final_result, temp_vision_frame.type());
    } else {
        final_result = result;
    }

    return final_result;
}

// ============================ GPU fused version ============================
PasteBackGPU::~PasteBackGPU() {
    if (d_temp_)   cudaFree(d_temp_);
    if (d_out_)    cudaFree(d_out_);
    if (d_crop_)   cudaFree(d_crop_);
    if (d_mask_)   cudaFree(d_mask_);
    if (d_affine_) cudaFree(d_affine_);
    if (h_temp_pinned_) cudaFreeHost(h_temp_pinned_);
    if (h_out_pinned_)  cudaFreeHost(h_out_pinned_);
}

void PasteBackGPU::ensure_capacity(size_t temp_bytes, size_t crop_bytes,
                                   size_t mask_bytes, size_t out_bytes) {
    if (temp_bytes > cap_temp_) {
        if (d_temp_) cudaFree(d_temp_);
        if (h_temp_pinned_) cudaFreeHost(h_temp_pinned_);
        cudaMalloc(&d_temp_, temp_bytes);
        cudaMallocHost(&h_temp_pinned_, temp_bytes);
        cap_temp_ = temp_bytes;
    }
    if (out_bytes > cap_out_) {
        if (d_out_) cudaFree(d_out_);
        if (h_out_pinned_) cudaFreeHost(h_out_pinned_);
        cudaMalloc(&d_out_, out_bytes);
        cudaMallocHost(&h_out_pinned_, out_bytes);
        cap_out_ = out_bytes;
    }
    if (crop_bytes > cap_crop_) {
        if (d_crop_) cudaFree(d_crop_);
        cudaMalloc(&d_crop_, crop_bytes);
        cap_crop_ = crop_bytes;
    }
    if (mask_bytes > cap_mask_) {
        if (d_mask_) cudaFree(d_mask_);
        cudaMalloc(&d_mask_, mask_bytes);
        cap_mask_ = mask_bytes;
    }
    if (d_affine_ == nullptr) cudaMalloc(&d_affine_, 6 * sizeof(float));
}

cv::Mat PasteBackGPU::paste_back(const cv::Mat& temp_vision_frame,
                                 const cv::Mat& crop_vision_frame,
                                 const cv::Mat& crop_mask,
                                 const cv::Mat& affine_matrix,
                                 cudaStream_t stream) {
    // normalize temp to a contiguous BGR uint8 frame
    cv::Mat temp = temp_vision_frame;
    if (temp.type() != CV_8UC3) temp.convertTo(temp, CV_8UC3);
    if (!temp.isContinuous()) temp = temp.clone();

    cv::Mat crop = crop_vision_frame.isContinuous() ? crop_vision_frame : crop_vision_frame.clone();
    cv::Mat mask = crop_mask.isContinuous() ? crop_mask : crop_mask.clone();

    const int W = temp.cols, H = temp.rows;
    const int Cw = crop.cols, Ch = crop.rows;
    const size_t temp_bytes = static_cast<size_t>(W) * H * 3;
    const size_t out_bytes  = temp_bytes;
    const size_t crop_bytes = static_cast<size_t>(Cw) * Ch * 3 * sizeof(float);
    const size_t mask_bytes = static_cast<size_t>(Cw) * Ch * sizeof(float);

    ensure_capacity(temp_bytes, crop_bytes, mask_bytes, out_bytes);

    // affine -> float[6] (estimateAffinePartial2D usually returns CV_64F)
    cv::Mat M64;
    affine_matrix.convertTo(M64, CV_64F);
    float h_aff[6];
    for (int i = 0; i < 6; ++i) h_aff[i] = static_cast<float>(M64.at<double>(i / 3, i % 3));

    // H2D (temp goes through a pinned staging buffer for true async transfer)
    std::memcpy(h_temp_pinned_, temp.data, temp_bytes);
    cudaMemcpyAsync(d_temp_, h_temp_pinned_, temp_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_crop_, crop.ptr<float>(), crop_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_mask_, mask.ptr<float>(), mask_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_affine_, h_aff, 6 * sizeof(float), cudaMemcpyHostToDevice, stream);

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    paste_back_fused_kernel<<<grid, block, 0, stream>>>(
        d_temp_, d_crop_, d_mask_, d_affine_, d_out_, W, H, Cw, Ch);

    cudaMemcpyAsync(h_out_pinned_, d_out_, out_bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cv::Mat result(H, W, CV_8UC3);
    std::memcpy(result.data, h_out_pinned_, out_bytes);
    return result;
}
