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
            // Upload the input frame to the device ONCE; both the NPP warp (here) and the
            // paste_back (below) read it straight from device memory — no second full-frame H2D.
            LITE_CPU_SCOPE_OPT(prof, "  upload");
            input_frame_.upload(face_swap_image, stream);
        }
        {
            // GPU NPP warp (reads the device-resident frame) -> device crop, then fused
            // bgr2rgb+normalize+HWC->CHW straight into the inference input buffer (buffers[0]).
            LITE_CPU_SCOPE_OPT(prof, "  warp+to_chw(gpu)");
            const unsigned char *d_crop = warp_npp_.warp_device_to_device(
                    input_frame_.data(), input_frame_.width(), input_frame_.height(),
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

    // ---------------- postprocess: transpose kernel + cvtColor + paste_back + blend ----------------
    cv::Mat dst_image;
    {
        LITE_CPU_SCOPE_OPT(prof, "postprocess");
        const int height = 512, width = 512;
        std::vector<float> transposed_data(1 * 3 * 512 * 512);
        {
            // GPU kernel writes HWC, BGR, float[0,255] straight into transposed_data,
            // folding the old CPU uint8->float conversion + cv::cvtColor(RGB2BGR).
            LITE_CPU_SCOPE_OPT(prof, "  transpose+dl");
            launch_face_restoration_postprocess(
                    static_cast<float *>(buffers[1]), transposed_data.data(), 3, 512, 512);
        }
        // aliases transposed_data (alive for the rest of this scope, i.e. through paste_back)
        cv::Mat mat(height, width, CV_32FC3, transposed_data.data());

        cv::Mat paste_frame;
        {
            // GPU fused: inverse-mapping sampling + blend in one kernel. temp frame is read
            // straight from the device-resident input_frame_ (no full-frame H2D here).
            LITE_CPU_SCOPE_OPT(prof, "  paste_back");
            paste_frame = paste_back_gpu_.paste_back(
                    input_frame_.data(), input_frame_.width(), input_frame_.height(),
                    mat, box_mask, affine_matrix, stream);
        }
        {
            LITE_CPU_SCOPE_OPT(prof, "  blend");
            dst_image = face_utils::blend_frame(ori_image, paste_frame);
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
