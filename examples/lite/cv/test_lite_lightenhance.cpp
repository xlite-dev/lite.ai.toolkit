//
// Created by wangzijian on 10/22/24.
//

#include "lite/lite.h"


static void test_default()
{
    std::string onnx_path = "../../../examples/hub/onnx/cv/diffusion_low_light_1x3x480x640.onnx";
    std::string test_img_path = "../../../examples/lite/resources/test_lite_light_enhance.png";
    std::string save_img_path = "../../../examples/logs/test_lite_light_enhance_onnx.jpg";

    // 1. Test Default Engine ONNXRuntime
    lite::cv::lightenhance::LightEnhance *light_enhance = new lite::cv::lightenhance::LightEnhance(onnx_path);

    cv::Mat img_bgr = cv::imread(test_img_path);

    light_enhance->detect(img_bgr, save_img_path);

    std::cout<<"light enhance done!"<<std::endl;

    delete light_enhance;

}

static void test_tensorrt()
{
#ifdef ENABLE_TENSORRT
    std::string engine_path = "../../../examples/hub/trt/diffusion_low_light_1x3x480x640.engine";
    std::string test_img_path = "../../../examples/lite/resources/test_lite_light_enhance.png";
    std::string save_img_path = "../../../examples/logs/test_lite_light_enhance_trt.jpg";

    lite::trt::cv::lightenhance::LightEnhance *lightenhance = new lite::trt::cv::lightenhance::LightEnhance (engine_path);

    cv::Mat test_image = cv::imread(test_img_path);

    lightenhance->detect(test_image,save_img_path);

    std::cout<<"trt light enhance done!"<<std::endl;

    delete lightenhance;
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
