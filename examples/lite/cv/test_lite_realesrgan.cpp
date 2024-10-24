//
// Created by wangzijian on 10/24/24.
//
#include "lite/lite.h"

static void test_default()
{
    std::string onnx_path = "../../../examples/hub/onnx/cv/RealESRGAN_x4plus.onnx";
    std::string test_img_path = "../../../examples/lite/resources/test_lite_realesrgan.jpg";
    std::string save_img_path = "../../../examples/logs/test_lite_realesrgan.jpg";

    // 1. Test Default Engine ONNXRuntime
    lite::cv::upscale::RealESRGAN *realEsrgan = new lite::cv::upscale::RealESRGAN (onnx_path);

    cv::Mat img_bgr = cv::imread(test_img_path);

    realEsrgan->detect(img_bgr, save_img_path);

    delete realEsrgan;
}


static void test_lite()
{
    test_default();
}

int main(__unused int argc, __unused char *argv[])
{
    test_lite();
    return 0;
}
