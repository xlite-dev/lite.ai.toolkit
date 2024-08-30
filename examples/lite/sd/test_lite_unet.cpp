//
// Created by wangzijian on 8/19/24.
//

#include "lite/lite.h"
#include "chrono"

static void test_default()
{
    auto start = std::chrono::high_resolution_clock::now();

    std::string engine_path = "/home/lite.ai.toolkit/examples/hub/onnx/sd/unet_model.onnx";

    lite::onnxruntime::sd::denoise::UNet *unet = new lite::onnxruntime::sd::denoise::UNet(engine_path);



    std::vector<std::string> input_vector = {"i am not good at cpp","goi ofg go !"};

    std::vector<std::vector<float>> output;

    unet->inference(input_vector,output);


    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "耗时: " << duration << " 毫秒" << std::endl;


    delete unet;

}


static void test_trt_unet()
{

    std::string engine_path = "/home/lite.ai.toolkit/examples/hub/trt/unet_fp16.engine";

    lite::trt::sd::denoise::UNet *unet = new lite::trt::sd::denoise::UNet(engine_path);

    for (int i = 0; i < 10; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        unet->inference();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "耗时: " << duration << " 毫秒" << std::endl;
    }



    delete unet;
}


static void test_lite()
{
    test_trt_unet();
//    test_default();
}

int main(__unused int argc, __unused char *argv[])
{
    test_lite();
    return 0;
}