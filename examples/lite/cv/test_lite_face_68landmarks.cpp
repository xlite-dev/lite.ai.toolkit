//
// Created by wangzijian on 11/1/24.
//
#include "lite/lite.h"
#include "lite/trt/cv/trt_face_68landmarks_mt.h"

static void test_default()
{
#ifdef ENABLE_ONNXRUNTIME
    std::string onnx_path = "/home/lite.ai.toolkit/examples/hub/onnx/cv/2dfan4.onnx";
    std::string test_img_path = "/home/lite.ai.toolkit/examples/lite/resources/test_lite_facefusion_pipeline_source.jpg";

    // 1. Test Default Engine ONNXRuntime
    lite::cv::faceid::Face_68Landmarks *face68Landmarks = new lite::cv::faceid::Face_68Landmarks(onnx_path);

    lite::types::BoundingBoxType<float, float> bbox;
    bbox.x1 = 487;
    bbox.y1 = 236;
    bbox.x2 = 784;
    bbox.y2 = 624;

    cv::Mat img_bgr = cv::imread(test_img_path);
    std::vector<cv::Point2f> face_landmark_5of68;
    face68Landmarks->detect(img_bgr, bbox, face_landmark_5of68);

    std::cout<<"face id detect done!"<<std::endl;

    delete face68Landmarks;
#endif
}




static void test_tensorrt()
{
#ifdef ENABLE_TENSORRT
    std::string engine_path = "/home/lite.ai.toolkit/examples/hub/trt/2dfan4_fp16.engine";
    std::string test_img_path = "/home/lite.ai.toolkit/1.jpg";

    // 1. Test TensorRT Engine
    lite::trt::cv::faceid::FaceFusionFace68Landmarks  *face68Landmarks = new lite::trt::cv::faceid::FaceFusionFace68Landmarks(engine_path);
    lite::types::BoundingBoxType<float, float> bbox;
    bbox.x1 = 487;
    bbox.y1 = 236;
    bbox.x2 = 784;
    bbox.y2 = 624;

    cv::Mat img_bgr = cv::imread(test_img_path);
    std::vector<cv::Point2f> face_landmark_5of68;
    face68Landmarks->detect(img_bgr, bbox, face_landmark_5of68);

    std::cout<<"face id detect done!"<<std::endl;

    delete face68Landmarks;
#endif
}


static void test_tensorrt_mt()
{
#ifdef ENABLE_TENSORRT
    std::string engine_path = "/home/lite.ai.toolkit/examples/hub/trt/2dfan4_fp16.engine";
    std::string test_img_path = "/home/lite.ai.toolkit/1.jpg";

    // 1. Test TensorRT Engine
//    lite::trt::cv::faceid::FaceFusionFace68Landmarks  *face68Landmarks = new lite::trt::cv::faceid::FaceFusionFace68Landmarks(engine_path);
    trt_face_68landmarks_mt *face68Landmarks = new trt_face_68landmarks_mt(engine_path,4);

    lite::types::BoundingBoxType<float, float> bbox;

    bbox.x1 = 487;
    bbox.y1 = 236;
    bbox.x2 = 784;
    bbox.y2 = 624;

    cv::Mat img_bgr = cv::imread(test_img_path);
    std::vector<cv::Point2f> face_landmark_5of68;
    face68Landmarks->detect_async(img_bgr, bbox, face_landmark_5of68);

    cv::Mat img_bgr2 = cv::imread(test_img_path);
    std::vector<cv::Point2f> face_landmark_5of682;
    face68Landmarks->detect_async(img_bgr, bbox, face_landmark_5of682);

    cv::Mat img_bgr3 = cv::imread(test_img_path);
    std::vector<cv::Point2f> face_landmark_5of683;
    face68Landmarks->detect_async(img_bgr, bbox, face_landmark_5of683);


    cv::Mat img_bgr4 = cv::imread(test_img_path);
    std::vector<cv::Point2f> face_landmark_5of684;
    face68Landmarks->detect_async(img_bgr, bbox, face_landmark_5of684);

    face68Landmarks->wait_for_completion();

    face68Landmarks->shutdown();

    std::cout<<"face id detect done!"<<std::endl;

    delete face68Landmarks;
#endif
}



int main(__unused int argc, __unused char *argv[])
{
//    test_tensorrt();
    test_tensorrt_mt();
//    test_default();
    return 0;
}