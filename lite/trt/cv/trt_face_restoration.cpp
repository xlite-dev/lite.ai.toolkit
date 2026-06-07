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
    // upload the input frame to the device once, then run the shared device-resident body.
    {
        LITE_CPU_SCOPE_OPT(prof, "  upload");
        input_frame_.upload(face_swap_image, stream);
    }
    return restore_core(input_frame_, target_landmarks_5, prof);
}

cv::Mat TRTFaceFusionFaceRestoration::restore(const DeviceFrame &input_frame,
                                              std::vector<cv::Point2f> &target_landmarks_5,
                                              lite::bench::Profiler *prof) {
    // input frame is already on the device (e.g. swap's output) — no upload.
    return restore_core(input_frame, target_landmarks_5, prof);
}

cv::Mat TRTFaceFusionFaceRestoration::restore_core(const DeviceFrame &frame,
                                                   std::vector<cv::Point2f> &target_landmarks_5,
                                                   lite::bench::Profiler *prof) {
    cv::Mat affine_matrix;
    cv::Mat box_mask;

    // ---------------- preprocess: estimate affine (CPU) -> GPU warp (NPP) -> fused CHW tensor ------
    {
        LITE_CPU_SCOPE_OPT(prof, "preprocess");
        {
            // Only the affine estimate stays on the CPU; the warp itself runs on the GPU and the
            // warped 512 crop stays device-resident (no D2H/H2D round-trip for the crop).
            LITE_CPU_SCOPE_OPT(prof, "  estimate_affine");
            affine_matrix = face_utils::estimate_affine_by_landmark_5(
                    target_landmarks_5, face_utils::FFHQ_512);
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
            // GPU NPP warp (reads the device-resident frame) -> device crop, then fused
            // bgr2rgb+normalize+HWC->CHW straight into the inference input buffer (buffers[0]).
            LITE_CPU_SCOPE_OPT(prof, "  warp+to_chw(gpu)");
            const unsigned char *d_crop = warp_npp_.warp_device_to_device(
                    frame.data(), frame.width(), frame.height(),
                    affine_matrix, 512, stream);
            preprocess_gpu_.run_device(d_crop, 512, 512, static_cast<float *>(buffers[0]), stream);
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

    // ---------------- postprocess: GPU transpose -> device crop -> paste+blend (all device) -------
    cv::Mat dst_image;
    {
        LITE_CPU_SCOPE_OPT(prof, "postprocess");
        const float *d_crop = nullptr;
        {
            // GPU kernel writes HWC BGR float[0,255] into a reusable DEVICE buffer (no D2H, no
            // per-call cudaMalloc) — fed straight to paste-back, so the 512 crop never hits host.
            LITE_CPU_SCOPE_OPT(prof, "  transpose(gpu)");
            d_crop = postproc_gpu_.run(static_cast<float *>(buffers[1]), 3, 512, 512, stream);
        }
        {
            // GPU fused: inverse-mapping sampling + paste + face-enhancer blend in ONE kernel.
            // Both the temp frame and the crop are device-resident; only the box mask is H2D'd.
            // blend_alpha=0.8 folds the old CPU blend_frame(target 0.2 / paste 0.8) in.
            LITE_CPU_SCOPE_OPT(prof, "  paste_back+blend");
            dst_image = paste_back_gpu_.paste_back(
                    frame.data(), frame.width(), frame.height(),
                    d_crop, 512, 512, box_mask, affine_matrix, stream, /*blend_alpha=*/0.8f);
        }
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
