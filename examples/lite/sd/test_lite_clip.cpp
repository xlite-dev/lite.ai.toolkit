//
// Created by wangzijian on 8/5/24.
//

#include "lite/lite.h"

static void test_default()
{
    std::string onnx_path = "../../../examples/hub/onnx/sd/clip_text_model_vitb32.onnx";

    lite::onnxruntime::sd::text_encoder::Clip *clip = new lite::onnxruntime::sd::text_encoder::Clip(onnx_path);

    std::vector<std::string> input_vector = {"i am not good at cpp","goi ofg go !"};

    std::vector<std::vector<float>> output;

    clip->inference(input_vector,output);

    delete clip;

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