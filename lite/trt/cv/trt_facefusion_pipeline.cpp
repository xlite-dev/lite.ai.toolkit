//
// Created by wangzijian on 11/14/24.
//

#include "trt_facefusion_pipeline.h"
#include <filesystem>
#include <stdexcept>
#include <string>
using trtcv::TRTFaceFusionPipeLine;

namespace {
// Fail fast with a clear message instead of crashing deep inside TensorRT/OpenCV.
inline void require_file(const std::string &path, const char *what) {
    if (!std::filesystem::exists(path))
        throw std::runtime_error(std::string("[FaceFusion] ") + what + " not found: " + path);
}
}

TRTFaceFusionPipeLine::TRTFaceFusionPipeLine(const std::string &face_detect_engine_path,
                                             const std::string &face_landmarks_68_engine_path,
                                             const std::string &face_recognizer_engine_path,
                                             const std::string &face_swap_engine_path,
                                             const std::string &face_restoration_engine_path) {
    require_file(face_detect_engine_path,       "face-detect engine");
    require_file(face_landmarks_68_engine_path, "face-landmarks engine");
    require_file(face_recognizer_engine_path,   "face-recognizer engine");
    require_file(face_swap_engine_path,         "face-swap engine");
    require_file(face_restoration_engine_path,  "face-restoration engine");

    face_detect  = std::make_unique<TRTYoloFaceV8>(face_detect_engine_path,1);
    face_landmarks = std::make_unique<TRTFaceFusionFace68Landmarks>(face_landmarks_68_engine_path,1);
    face_recognizer = std::make_unique<TRTFaceFusionFaceRecognizer>(face_recognizer_engine_path,1);
    face_swap = std::make_unique<TRTFaceFusionFaceSwap>(face_swap_engine_path,1);
    face_restoration = std::make_unique<TRTFaceFusionFaceRestoration>(face_restoration_engine_path,1);
}

// Run the SOURCE branch once (detect -> landmark -> recognize) and cache its embedding,
// so per-frame process() only has to handle the target. NO disk I/O. Per-stage timing is
// opt-in via prof (LITE_CPU_SCOPE_OPT is zero-overhead when null).
void TRTFaceFusionPipeLine::prepare_source(const cv::Mat &source_image, int src_index,
                                           lite::bench::Profiler *prof) {
    if (source_image.empty())
        throw std::runtime_error("[FaceFusion] source image is empty");

    cv::Mat img_bgr = source_image.clone();   // sub-models take a non-const cv::Mat&
    cv::Mat img_bgr_src = img_bgr.clone();

    std::vector<lite::types::Boxf> detected_boxes;
    { LITE_CPU_SCOPE_OPT(prof, "detect_src");
      face_detect->detect(img_bgr, detected_boxes, 0.25f, 0.45f); }

    std::vector<lite::types::Boxf> src_final_boxes;
    for (auto current_box : detected_boxes)
        if (current_box.score != 0) src_final_boxes.emplace_back(current_box);

    if (src_final_boxes.empty())
        throw std::runtime_error("[FaceFusion] no face detected in source image");
    if (src_index < 0 || src_index >= static_cast<int>(src_final_boxes.size()))
        throw std::runtime_error("[FaceFusion] source face index " + std::to_string(src_index) +
                                 " out of range (" + std::to_string(src_final_boxes.size()) + " face(s) detected)");

    std::vector<cv::Point2f> face_landmark_5of68;
    int src_pick = (src_final_boxes.size() == 1) ? 0 : src_index;
    { LITE_CPU_SCOPE_OPT(prof, "landmark_src");
      face_landmarks->detect(img_bgr, src_final_boxes[src_pick], face_landmark_5of68); }

    { LITE_CPU_SCOPE_OPT(prof, "recognizer");
      face_recognizer->detect(img_bgr_src, face_landmark_5of68, source_embedding_); }
    source_ready_ = true;
}

// Per target frame: detect + landmark on the target, swap the cached source face, restore.
// NO disk I/O. Requires a prior prepare_source().
cv::Mat TRTFaceFusionPipeLine::process(const cv::Mat &target_image, int target_index,
                                       lite::bench::Profiler *prof) {
    if (!source_ready_)
        throw std::runtime_error("[FaceFusion] process() called before prepare_source()");
    if (target_image.empty())
        throw std::runtime_error("[FaceFusion] target image is empty");

    cv::Mat target_img_bgr = target_image.clone();

    std::vector<lite::types::Boxf> target_detected_boxes;
    { LITE_CPU_SCOPE_OPT(prof, "detect_tgt");
      face_detect->detect(target_img_bgr, target_detected_boxes, 0.25f, 0.45f); }

    std::vector<lite::types::Boxf> target_final_boxes;
    for (auto current_box : target_detected_boxes)
        if (current_box.score != 0) target_final_boxes.emplace_back(current_box);

    if (target_final_boxes.empty())
        throw std::runtime_error("[FaceFusion] no face detected in target image");
    if (target_index < 0 || target_index >= static_cast<int>(target_final_boxes.size()))
        throw std::runtime_error("[FaceFusion] target face index " + std::to_string(target_index) +
                                 " out of range (" + std::to_string(target_final_boxes.size()) + " face(s) detected)");

    std::vector<cv::Point2f> target_face_landmark_5of68;
    int tgt_pick = (target_final_boxes.size() == 1) ? 0 : target_index;
    { LITE_CPU_SCOPE_OPT(prof, "landmark_tgt");
      face_landmarks->detect(target_img_bgr, target_final_boxes[tgt_pick], target_face_landmark_5of68); }

    // ---- swap + restore: the swapped frame stays GPU-resident in swapped_frame_, so restoration
    //      reads it from the device (no swap-D2H + restoration-H2D round-trip across the seam) ----
    { LITE_CPU_SCOPE_OPT(prof, "swap");
      face_swap->detect(target_img_bgr, source_embedding_, target_face_landmark_5of68, swapped_frame_); }

    cv::Mat result;
    { LITE_CPU_SCOPE_OPT(prof, "restoration");
      result = face_restoration->restore(swapped_frame_, target_face_landmark_5of68, nullptr); }
    return result;
}

// Convenience one-shot: prepare the source then process the target (recomputes the source
// embedding on every call — for video, call prepare_source() once and process() per frame).
cv::Mat TRTFaceFusionPipeLine::detect(const cv::Mat &source_image, int src_index,
                                      const cv::Mat &target_image, int target_index,
                                      lite::bench::Profiler *prof) {
    prepare_source(source_image, src_index, prof);
    return process(target_image, target_index, prof);
}

// Convenience wrapper: file paths in, result written to disk. Thin layer over the
// in-memory core; imread/imwrite are timed separately when a Profiler is passed.
void TRTFaceFusionPipeLine::detect(const std::string &source_image, int src_index,
                                   const std::string &target_image, int target_index,
                                   const std::string &save_image,
                                   lite::bench::Profiler *prof) {
    cv::Mat src, tgt;
    { LITE_CPU_SCOPE_OPT(prof, "imread_src"); src = cv::imread(source_image); }
    if (src.empty())
        throw std::runtime_error("[FaceFusion] cannot read source image: " + source_image);
    { LITE_CPU_SCOPE_OPT(prof, "imread_tgt"); tgt = cv::imread(target_image); }
    if (tgt.empty())
        throw std::runtime_error("[FaceFusion] cannot read target image: " + target_image);

    cv::Mat out = detect(src, src_index, tgt, target_index, prof);

    { LITE_CPU_SCOPE_OPT(prof, "imwrite"); cv::imwrite(save_image, out); }
}
