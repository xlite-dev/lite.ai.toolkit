//
// Created by wangzijian on 8/19/24.
//

#include "lite/lite.h"


static void test_default()
{
    std::string engine_path = "/home/lite.ai.toolkit/examples/hub/onnx/sd/unet_model.onnx";

    lite::onnxruntime::sd::denoise::UNet *unet = new lite::onnxruntime::sd::denoise::UNet(engine_path);



    std::vector<std::string> input_vector = {"i am not good at cpp","goi ofg go !"};

    std::vector<std::vector<float>> output;

    unet->inference(input_vector,output);

    delete unet;

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