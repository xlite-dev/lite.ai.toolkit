//
// Created by wangzijian on 11/7/24.
//
#include "lite/lite.h"

static void test_default()
{
#ifdef ENABLE_ONNXRUNTIME
    std::string onnx_path = "../../../examples/hub/onnx/cv/gfpgan_1.4.onnx";
    std::string test_img_path = "../../../examples/lite/resources/test_lite_facefusion_pipeline_source.jpg";
    std::string save_img_path = "../../../examples/lite/resources/test_lite_facefusion_pipeline_target.jpg";

    // 1. Test Default Engine ONNXRuntime
    lite::cv::face::restoration::GFPGAN *face_restoration = new  lite::cv::face::restoration::GFPGAN(onnx_path);

    std::vector<cv::Point2f> face_landmark_5 = {
            cv::Point2f(569.092041f, 398.845886f),
            cv::Point2f(701.891724f, 399.156677f),
            cv::Point2f(634.767212f, 482.927216f),
            cv::Point2f(584.270996f, 543.294617f),
            cv::Point2f(684.877991f, 543.067078f)
    };
    cv::Mat img_bgr = cv::imread(test_img_path);

    face_restoration->detect(img_bgr,face_landmark_5,save_img_path);


    std::cout<<"face restoration detect done!"<<std::endl;

    delete face_restoration;
#endif
}

int main(__unused int argc, __unused char *argv[])
{
    test_default();
    return 0;
}