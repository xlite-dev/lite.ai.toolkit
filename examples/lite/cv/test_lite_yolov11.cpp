//
// Created by wangizijian on 24-7-8.
//

#include "lite/lite.h"



static void test_tensorrt()
{
#ifdef ENABLE_TENSORRT
    std::string engine_path = "/root/autodl-tmp/lite.ai.toolkit/examples/hub/onnx/cv/yolo11_fp32.engine";
    std::string test_img_path = "/root/autodl-tmp/lite.ai.toolkit/examples/lite/resources/test_lite_yolov5_2.jpg";
    std::string save_img_path = "/root/autodl-tmp/lite.ai.toolkit/examples/test_lite_yolov8_trt_1.jpg";

    lite::trt::cv::detection::YOLOV11 *yolov11  = new lite::trt::cv::detection::YOLOV11(engine_path);

    cv::Mat test_image = cv::imread(test_img_path);

    std::vector<lite::types::Boxf> detected_boxes;

    yolov11->detect(test_image,detected_boxes,0.5f,0.4f);

    std::cout<<"trt yolov8 detect done!"<<std::endl;
    lite::utils::draw_boxes_inplace(test_image, detected_boxes);
    cv::imwrite(save_img_path, test_image);

    delete yolov11;
#endif
}

static void test_lite()
{
    test_tensorrt();
}



int main(__unused int argc, __unused char *argv[])
{
    test_lite();
    return 0;
}
