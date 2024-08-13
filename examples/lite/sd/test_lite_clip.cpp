//
// Created by wangzijian on 8/5/24.
//

#include "lite/lite.h"
#include "lite/ort/sd/ddimscheduler.h"

static void test_default()
{
    std::string onnx_path = "/home/lite.ai.toolkit/examples/hub/onnx/sd/clip_text_model_vitb32.onnx";

    auto scheduler = Scheduler::DDIMScheduler("/home/lite.ai.toolkit/lite/ort/sd/scheduler_config.json");
    scheduler.set_timesteps(30);
    std::vector<int> timesteps;
    scheduler.get_timesteps(timesteps);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::normal_distribution<float> distribution(.0, 1.0);

    std::vector<float> sample(1 * 4 * 64 * 64);
    std::vector<float> model_output(1 * 4 * 64 * 64);

    for(int i = 0; i < 4 * 64 * 64; i++){
        sample[i] = distribution(generator);
        model_output[i] = distribution(generator);
    }

    std::vector<float> pred_sample;

    for(auto t: timesteps){
        scheduler.step(model_output, {1, 4, 3, 3}, sample, {1, 4, 3, 3}, pred_sample, t);
    }
    std::cout << "passed!" << std::endl;

    lite::onnxruntime::sd::text_encoder::Clip *clip = new lite::onnxruntime::sd::text_encoder::Clip(onnx_path);

    std::vector<std::string> input_vector = {"i am not good at cpp","goi ofg go !"};

    std::vector<std::vector<float>> output;

    clip->inference(input_vector,output);

    delete clip;

}


static void test_tensorrt()
{
    std::string engine_path = "../../../examples/hub/trt/dynamic_text_model_fp32.engine";

    lite::trt::sd::text_encoder::Clip *clip = new lite::trt::sd::text_encoder::Clip(engine_path);

    std::vector<std::string> input_vector = {"i am not good at cpp","goi ofg go !"};

    std::vector<std::vector<float>> output;

    clip->inference(input_vector,output);

    delete clip;

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