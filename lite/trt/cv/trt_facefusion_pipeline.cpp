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

// Per-stage timing is opt-in: pass a Profiler to break the pipeline down into
// imread / detect / landmark (x2, source+target) / recognizer / swap / restoration.
// When prof == nullptr the LITE_CPU_SCOPE_OPT scopes are zero-overhead. Each stage
// returns a host-visible result (boxes / landmarks / Mat), so it synchronizes and
// CPU-side wall-clock timing is accurate.
void TRTFaceFusionPipeLine::detect(const std::string &source_image, int src_index,
                                   const std::string &target_image, int target_index,
                                   const std::string &save_image,
                                   lite::bench::Profiler *prof) {
    // ---- source: image -> detect -> landmarks -> recognizer -> embedding ----
    cv::Mat img_bgr;
    { LITE_CPU_SCOPE_OPT(prof, "imread_src"); img_bgr = cv::imread(source_image); }
    if (img_bgr.empty())
        throw std::runtime_error("[FaceFusion] cannot read source image: " + source_image);
    auto img_bgr_src = img_bgr.clone();

    std::vector<lite::types::Boxf> detected_boxes;
    { LITE_CPU_SCOPE_OPT(prof, "detect_src");
      face_detect->detect(img_bgr, detected_boxes, 0.25f, 0.45f); }

    std::vector<lite::types::Boxf> src_final_boxes;
    for (auto current_box : detected_boxes)
        if (current_box.score != 0) src_final_boxes.emplace_back(current_box);

    if (src_final_boxes.empty())
        throw std::runtime_error("[FaceFusion] no face detected in source image: " + source_image);
    if (src_index < 0 || src_index >= static_cast<int>(src_final_boxes.size()))
        throw std::runtime_error("[FaceFusion] source face index " + std::to_string(src_index) +
                                 " out of range (" + std::to_string(src_final_boxes.size()) + " face(s) detected)");

    std::vector<cv::Point2f> face_landmark_5of68;
    int src_pick = (src_final_boxes.size() == 1) ? 0 : src_index;
    { LITE_CPU_SCOPE_OPT(prof, "landmark_src");
      face_landmarks->detect(img_bgr, src_final_boxes[src_pick], face_landmark_5of68); }

    std::vector<float> source_image_embeding;
    { LITE_CPU_SCOPE_OPT(prof, "recognizer");
      face_recognizer->detect(img_bgr_src, face_landmark_5of68, source_image_embeding); }

    // ---- target: image -> detect -> landmarks ----
    cv::Mat target_img_bgr;
    { LITE_CPU_SCOPE_OPT(prof, "imread_tgt"); target_img_bgr = cv::imread(target_image); }
    if (target_img_bgr.empty())
        throw std::runtime_error("[FaceFusion] cannot read target image: " + target_image);

    std::vector<lite::types::Boxf> target_detected_boxes;
    { LITE_CPU_SCOPE_OPT(prof, "detect_tgt");
      face_detect->detect(target_img_bgr, target_detected_boxes, 0.25f, 0.45f); }

    std::vector<lite::types::Boxf> target_final_boxes;
    for (auto current_box : target_detected_boxes)
        if (current_box.score != 0) target_final_boxes.emplace_back(current_box);

    if (target_final_boxes.empty())
        throw std::runtime_error("[FaceFusion] no face detected in target image: " + target_image);
    if (target_index < 0 || target_index >= static_cast<int>(target_final_boxes.size()))
        throw std::runtime_error("[FaceFusion] target face index " + std::to_string(target_index) +
                                 " out of range (" + std::to_string(target_final_boxes.size()) + " face(s) detected)");

    std::vector<cv::Point2f> target_face_landmark_5of68;
    int tgt_pick = (target_final_boxes.size() == 1) ? 0 : target_index;
    { LITE_CPU_SCOPE_OPT(prof, "landmark_tgt");
      face_landmarks->detect(target_img_bgr, target_final_boxes[tgt_pick], target_face_landmark_5of68); }

    // ---- swap + restore ----
    cv::Mat face_swap_image;
    { LITE_CPU_SCOPE_OPT(prof, "swap");
      face_swap->detect(target_img_bgr, source_image_embeding, target_face_landmark_5of68, face_swap_image); }
    { LITE_CPU_SCOPE_OPT(prof, "restoration");
      face_restoration->detect(face_swap_image, target_face_landmark_5of68, save_image); }
}
