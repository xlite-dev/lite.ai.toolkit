//
// Created by wangzijian on 11/1/24.
//
#include "lite/lite.h"

static void test_default()
{
#ifdef ENABLE_ONNXRUNTIME
    std::string onnx_path = "/home/lite.ai.toolkit/examples/hub/onnx/cv/2dfan4.onnx";
    std::string test_img_path = "/home/lite.ai.toolkit/5.jpg";
    std::string save_img_path = "/home/lite.ai.toolkit/5_1.jpg";

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

int main(__unused int argc, __unused char *argv[])
{
    test_default();
    return 0;
}