//
// Created by wangzijian on 11/5/24.
//
#include "lite/lite.h"

static void test_default()
{
#ifdef ENABLE_ONNXRUNTIME
    std::string onnx_path = "../../../examples/hub/onnx/cv/arcface_w600k_r50.onnx";
    std::string test_img_path = "../../../examples/lite/resources/test_lite_facefusion_pipeline_source.jpg";

    // 1. Test Default Engine ONNXRuntime
    lite::cv::faceid::Face_Recognizer *face_recognizer = new lite::cv::faceid::Face_Recognizer(onnx_path);

    std::vector<cv::Point2f> face_landmark_5 = {
            cv::Point2f(568.2485f, 398.9512f),
            cv::Point2f(701.7346f, 399.64795f),
            cv::Point2f(634.2213f, 482.92694f),
            cv::Point2f(583.5656f, 543.10187f),
            cv::Point2f(684.52405f, 543.125f)
    };
    cv::Mat img_bgr = cv::imread(test_img_path);

    std::vector<float> source_image_embeding;
    
    face_recognizer->detect(img_bgr,face_landmark_5,source_image_embeding);


    std::cout<<"face id detect done!"<<std::endl;

    delete face_recognizer;
#endif
}

int main(__unused int argc, __unused char *argv[])
{
    test_default();
    return 0;
}