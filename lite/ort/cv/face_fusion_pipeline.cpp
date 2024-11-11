//
// Created by wangzijian on 11/7/24.
//

#include "face_fusion_pipeline.h"
using ortcv::Face_Fusion_Pipeline;

Face_Fusion_Pipeline::Face_Fusion_Pipeline(const std::string &face_detect_onnx_path,
                                           const std::string &face_landmarks_68_onnx_path,
                                           const std::string &face_recognizer_onnx_path,
                                           const std::string &face_swap_onnx_path,
                                           const std::string &face_restoration_onnx_path) {
    face_detect  = std::make_unique<YoloFaceV8>(face_detect_onnx_path,6);
    face_landmarks = std::make_unique<Face_68Landmarks>(face_landmarks_68_onnx_path,6);
    face_recognizer = std::make_unique<Face_Recognizer>(face_recognizer_onnx_path,6);
    face_swap = std::make_unique<Face_Swap>(face_swap_onnx_path,6);
    face_restoration = std::make_unique<Face_Restoration>(face_restoration_onnx_path,6);
}

void Face_Fusion_Pipeline::detect(const std::string &source_image, const std::string &target_image,const std::string &save_image_path) {
    std::vector<lite::types::Boxf> detected_boxes;
    cv::Mat img_bgr = cv::imread(source_image);
    face_detect->detect(img_bgr,detected_boxes);

    int position = 0; // position number 0
    auto test_bounding_box = detected_boxes[0];
    std::vector<cv::Point2f> face_landmark_5of68;

    face_landmarks->detect(img_bgr, test_bounding_box, face_landmark_5of68);
    std::vector<float> source_image_embeding;
    face_recognizer->detect(img_bgr,face_landmark_5of68,source_image_embeding);


    std::vector<lite::types::Boxf> target_detected_boxes;
    cv::Mat target_img_bgr = cv::imread(target_image);
    face_detect->detect(target_img_bgr, target_detected_boxes);
    auto target_test_bounding_box = target_detected_boxes[0];
    std::vector<cv::Point2f> target_face_landmark_5of68;
    face_landmarks->detect(target_img_bgr, target_test_bounding_box,target_face_landmark_5of68);

    cv::Mat face_swap_image;
    face_swap->detect(target_img_bgr,source_image_embeding,target_face_landmark_5of68,face_swap_image);
    face_restoration->detect(face_swap_image,target_face_landmark_5of68,save_image_path);

}