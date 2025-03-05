//
// Created by wangzijian on 2/26/25.
//
#include "lite/lite.h"


static void test_default()
{
    std::string onnx_path = "../../../examples/hub/onnx/cv/unetcnn.onnx";
    std::string test_img_path = "../../../examples/lite/resources/test_lite_docunwarp.jpg";
    std::string save_img_path = "../../../examples/logs/test_lite_age_docunwarp.jpg";

    // 1. Test Default Engine ONNXRuntime
    lite::cv::ocr::DocUnWarp *docUnWarp = new lite::cv::ocr::DocUnWarp(onnx_path);
    cv::Mat img_bgr = cv::imread(test_img_path);
    cv::Mat temp_image;
    docUnWarp->detect(img_bgr,temp_image);
    cv::imwrite(save_img_path,temp_image);
    std::cout<<"doc binary done!"<<std::endl;

    delete docUnWarp;

}



int main(__unused int argc, __unused char *argv[])
{
    test_default();
    return 0;
}