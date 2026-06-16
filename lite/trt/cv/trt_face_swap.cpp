//
// Created by wangzijian on 11/13/24.
//

#include "trt_face_swap.h"
using trtcv::TRTFaceFusionFaceSwap;

// infer + postprocess: produces the host BGR float[0,255] 128x128 swapped-face crop (owns its
// data) and sets affine_martix. Uploads the target frame ONCE into target_dev_ (shared by the
// warp here AND the caller's paste-back). Shared by both detect() overloads.
void TRTFaceFusionFaceSwap::swap_core(cv::Mat &target_image, std::vector<float> &source_face_embeding,
                                      std::vector<cv::Point2f> &target_landmark_5, cv::Mat &mat_out) {
    // upload the target frame to the device ONCE — warp (here) + paste-back (caller) both read it.
    target_dev_.upload(target_image, stream);

    // source embedding (CPU, cheap): model_matrix_ loaded once in ctor.
    std::vector<float> source_embeding_input = face_utils::dot_product(source_face_embeding, model_matrix_, 512);
    face_utils::normalize(source_embeding_input);

    // image: estimate the ARCFACE-128 affine (CPU) -> NPP warp the 128 crop FROM the device frame
    // -> fused bgr2rgb + /255 + HWC->CHW straight into buffers[0]. Replaces CPU warpAffine + cvtColor
    // + convertTo + create_tensor + the CHW-tensor H2D, with no extra full-frame upload (reuses
    // target_dev_). inswapper input is RGB normalized to [0,1].
    affine_martix = face_utils::estimate_affine_by_landmark_5(target_landmark_5, face_utils::ARCFACE_128_V2);
    const unsigned char* d_crop = warp_npp_.warp_device_to_device(
            target_dev_.data(), target_dev_.width(), target_dev_.height(), affine_martix, 128, stream);
    preprocess_gpu_.run_device(d_crop, 128, 128, static_cast<float*>(buffers[0]), stream,
                               /*scale=*/1.0f / 255.f, /*bias=*/0.0f);

    cudaMemcpyAsync(buffers[1],source_embeding_input.data(),512 * sizeof(float), cudaMemcpyHostToDevice,stream);
    cudaStreamSynchronize(stream);

    bool status = trt_context->enqueueV3(stream);
    if (!status) {
        std::cerr << "Failed to enqueue TensorRT model." << std::endl;
        return;
    }

    std::vector<float> output_vector(3 * 128 * 128);
    cudaMemcpyAsync(output_vector.data(),buffers[2],1 * 3 * 128 * 128 * sizeof(float),cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);

    // CHW float -> HWC uint8-range (denormalize); paste-back is done on the GPU by the caller.
    std::vector<float> transposed(3 * 128 * 128);
    const int channels = 3, height = 128, width = 128;
#pragma omp parallel for collapse(3)
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int src_idx = c * (height * width) + h * width + w;  // CHW
                int dst_idx = h * (width * channels) + w * channels + c;  // HWC
                transposed[dst_idx] = output_vector[src_idx];
            }
        }
    }
    for (auto& val : transposed) {
        val = std::round(val * 255.0);
    }

    cv::Mat mat(height, width, CV_32FC3, transposed.data());
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
    mat.copyTo(mat_out);   // own the data (transposed is local)
}

void TRTFaceFusionFaceSwap::detect(cv::Mat &target_image, std::vector<float> source_face_embeding,
                                   std::vector<cv::Point2f> target_landmark_5, cv::Mat &face_swap_image) {
    cv::Mat mat;
    swap_core(target_image, source_face_embeding, target_landmark_5, mat);
    // paste-back reads the temp frame from the shared device-resident target_dev_ (no extra H2D).
    face_swap_image = paste_back_gpu_.paste_back(
            target_dev_.data(), target_dev_.width(), target_dev_.height(),
            mat, box_mask_, affine_martix, stream);
}

void TRTFaceFusionFaceSwap::detect(cv::Mat &target_image, std::vector<float> source_face_embeding,
                                   std::vector<cv::Point2f> target_landmark_5, DeviceFrame &out_frame) {
    cv::Mat mat;
    swap_core(target_image, source_face_embeding, target_landmark_5, mat);
    // paste straight into the device-resident out_frame (no D2H), temp read from shared target_dev_.
    paste_back_gpu_.paste_back_to_device(
            target_dev_.data(), target_dev_.width(), target_dev_.height(),
            mat, box_mask_, affine_martix, out_frame, stream);
    // restoration reads out_frame on its OWN stream, so make sure this paste has completed.
    cudaStreamSynchronize(stream);
}
