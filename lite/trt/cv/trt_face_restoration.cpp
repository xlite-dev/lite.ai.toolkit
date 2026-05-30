//
// Created by wangzijian on 11/14/24.
//

#include "trt_face_restoration.h"
#include "lite/bench/profiler.h"
using trtcv::TRTFaceFusionFaceRestoration;

// Core compute path: returns the restored full frame, no disk write.
// When prof is non-null, records per-stage timings (preprocess / infer / postprocess,
// with paste_back broken out separately).
cv::Mat TRTFaceFusionFaceRestoration::restore(cv::Mat &face_swap_image,
                                              std::vector<cv::Point2f> &target_landmarks_5,
                                              lite::bench::Profiler *prof) {
    auto ori_image = face_swap_image.clone();

    cv::Mat crop_image;
    cv::Mat affine_matrix;
    cv::Mat box_mask;

    // ---------------- preprocess (CPU): warp + bgr2rgb + normalize + build tensor ----------------
    {
        LITE_CPU_SCOPE_OPT(prof, "preprocess");
        {
            LITE_CPU_SCOPE_OPT(prof, "  warp");
            std::tie(crop_image, affine_matrix) = face_utils::warp_face_by_face_landmark_5(
                    face_swap_image, target_landmarks_5, face_utils::FFHQ_512);
        }
        {
            // the static box mask only depends on the (fixed) 512 crop size, so build it
            // once and reuse — it used to be rebuilt every frame (a large-kernel GaussianBlur, ~10ms)
            LITE_CPU_SCOPE_OPT(prof, "  mask");
            if (box_mask_cache_.empty())
                box_mask_cache_ = face_utils::create_static_box_mask({512, 512});
            box_mask = box_mask_cache_;
        }

        {
            // GPU fused: bgr2rgb + normalize + HWC->CHW written straight into the inference
            // input buffer (buffers[0]) — also removes the separate H2D below.
            LITE_CPU_SCOPE_OPT(prof, "  to_chw(gpu)");
            preprocess_gpu_.run(crop_image, static_cast<float *>(buffers[0]), stream);
        }
    }

    // ---------------- inference (GPU + sync); input already in buffers[0] ----------------
    {
        LITE_CPU_SCOPE_OPT(prof, "infer(gpu)");
        bool status = trt_context->enqueueV3(stream);
        if (!status) {
            std::cerr << "Failed to inference" << std::endl;
            return cv::Mat();
        }
        cudaStreamSynchronize(stream);
    }

    // ---------------- postprocess: transpose kernel + cvtColor + paste_back + blend ----------------
    cv::Mat dst_image;
    {
        LITE_CPU_SCOPE_OPT(prof, "postprocess");
        std::vector<unsigned char> transposed_data(1 * 3 * 512 * 512);
        launch_face_restoration_postprocess(
                static_cast<float *>(buffers[1]), transposed_data.data(), 3, 512, 512);
        std::vector<float> transposed_data_float(transposed_data.begin(), transposed_data.end());
        cudaStreamSynchronize(stream);

        int height = 512, width = 512;
        cv::Mat mat(height, width, CV_32FC3, transposed_data_float.data());
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);

        cv::Mat paste_frame;
        {
            // GPU fused version: inverse-mapping sampling + blend in one kernel, reused
            // device buffers, pinned + async copies (replaces the CPU warpAffine bottleneck).
            LITE_CPU_SCOPE_OPT(prof, "  paste_back");
            paste_frame = paste_back_gpu_.paste_back(ori_image, mat, box_mask, affine_matrix, stream);
        }
        dst_image = face_utils::blend_frame(ori_image, paste_frame);
    }

    return dst_image;
}

void TRTFaceFusionFaceRestoration::detect(cv::Mat &face_swap_image,
                                          std::vector<cv::Point2f> &target_landmarks_5,
                                          const std::string &face_enchaner_path) {
    cv::Mat dst_image = restore(face_swap_image, target_landmarks_5, nullptr);
    if (!dst_image.empty())
        cv::imwrite(face_enchaner_path, dst_image);
}
