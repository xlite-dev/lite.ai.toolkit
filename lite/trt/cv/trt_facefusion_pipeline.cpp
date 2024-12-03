//
// Created by wangzijian on 11/14/24.
//

#include "trt_facefusion_pipeline.h"
using trtcv::TRTFaceFusionPipeLine;

TRTFaceFusionPipeLine::TRTFaceFusionPipeLine(const std::string &face_detect_engine_path,
                                             const std::string &face_landmarks_68_engine_path,
                                             const std::string &face_recognizer_engine_path,
                                             const std::string &face_swap_engine_path,
                                             const std::string &face_restoration_engine_path) {
    face_detect  = std::make_unique<TRTYoloFaceV8>(face_detect_engine_path,1);
    face_landmarks = std::make_unique<TRTFaceFusionFace68Landmarks>(face_landmarks_68_engine_path,1);
    face_recognizer = std::make_unique<TRTFaceFusionFaceRecognizer>(face_recognizer_engine_path,1);
    face_swap = std::make_unique<TRTFaceFusionFaceSwap>(face_swap_engine_path,1);
    face_restoration = std::make_unique<TRTFaceFusionFaceRestoration>(face_restoration_engine_path,1);


}

void TRTFaceFusionPipeLine::detect(const std::string &source_image, const std::string &target_image,
                                   const std::string &save_image) {
    // source 的全部流程
    // image -> detect -> landmarks -> recognizer -> embeding
    // 最终也就是 image -> embeding
    std::vector<lite::types::Boxf> detected_boxes;
    cv::Mat img_bgr = cv::imread(source_image);
    auto img_bgr_src = img_bgr.clone();
    face_detect->detect(img_bgr,detected_boxes,0.25f,0.45f);
    int position = 0; // position number 0
    auto test_bounding_box = detected_boxes[0];
    std::vector<cv::Point2f> face_landmark_5of68;
    face_landmarks->detect(img_bgr, test_bounding_box,face_landmark_5of68);

    // 这里准备使用多线程来进行操作 因为这里的操作和下面target的操作是独立的
    // 这段代码仅仅是为了测试多线程的效果
    // 到时候需要更改
//    std::string engine_path = "/home/lite.ai.toolkit/examples/hub/trt/2dfan4_fp16.engine";
//    trt_face_68landmarks_mt *face68Landmarks = new trt_face_68landmarks_mt(engine_path,2);
//    face68Landmarks->detect_async(img_bgr, test_bounding_box, face_landmark_5of68);
//    face68Landmarks->wait_for_completion();



//    face_landmarks->detect(img_bgr, test_bounding_box, face_landmark_5of68);
    std::vector<float> source_image_embeding;
    face_recognizer->detect(img_bgr_src,face_landmark_5of68,source_image_embeding);

    // target 的全部流程
    // image -> detect -> landmarks
    // 最终也就是 image -> landmarks
    std::vector<lite::types::Boxf> target_detected_boxes;
    cv::Mat target_img_bgr = cv::imread(target_image);
    auto target_img_bgr_src = target_img_bgr.clone();
    face_detect->detect(target_img_bgr, target_detected_boxes,0.25f,0.45f);
    auto target_test_bounding_box = target_detected_boxes[0];
    std::vector<cv::Point2f> target_face_landmark_5of68;
//    face68Landmarks->detect_async(target_img_bgr_src, target_test_bounding_box, target_face_landmark_5of68);
//    face68Landmarks->wait_for_completion();
//    face68Landmarks->shutdown();

    face_landmarks->detect(target_img_bgr, target_test_bounding_box,target_face_landmark_5of68);

    // 公共部分
    cv::Mat face_swap_image;
    face_swap->detect(target_img_bgr,source_image_embeding,target_face_landmark_5of68,face_swap_image);
    face_restoration->detect(face_swap_image,target_face_landmark_5of68,save_image);
}


