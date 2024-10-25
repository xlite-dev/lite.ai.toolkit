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


static void test_tensorrt()
{
#ifdef ENABLE_TENSORRT
    std::string engine_path = "../../../examples/hub/trt/RealESRGAN_x4plus_fp16.engine";
    std::string test_img_path = "../../../examples/lite/resources/test_lite_realesrgan.jpg";
    std::string save_img_path = "../../../examples/logs/test_lite_realesrgan_trt.jpg";

    lite::trt::cv::upscale::RealESRGAN *realesrgan = new lite::trt::cv::upscale::RealESRGAN (engine_path);

    cv::Mat test_image = cv::imread(test_img_path);

    realesrgan->detect(test_image,save_img_path);

    std::cout<<"trt upscale enhance done!"<<std::endl;

    delete realesrgan;
#endif
}


static void test_lite()
{
    test_default();
    test_tensorrt();
}

int main(__unused int argc, __unused char *argv[])
{
    test_lite();
    return 0;
}
