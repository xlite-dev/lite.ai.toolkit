//
// Created by wangzijian on 8/31/24.
//
#include "lite/lite.h"

static void test_default()
{
    std::string clip_onnx_path = "/home/wangzijian/lite.ai.toolkit/examples/hub/onnx/sd/clip_model.onnx";
    std::string unet_onnx_path = "/home/wangzijian/lite.ai.toolkit/examples/hub/onnx/sd/unet_model.onnx";
    std::string vae_onnx_path = "/home/wangzijian/lite.ai.toolkit/examples/hub/onnx/sd/vae_model.onnx";

    auto *pipeline = new lite::onnxruntime::sd::pipeline::Pipeline(clip_onnx_path, unet_onnx_path,
                                                                   vae_onnx_path,
                                                                   1);

    std::string prompt = "1girl with red hair,blue eyes,smile, looking at viewer";
    std::string negative_prompt = "";
    std::string save_path = "/home/wangzijian/lite.ai.toolkit/output_merge.png";
    pipeline->inference(prompt,negative_prompt,save_path);

    delete pipeline;

}


static void test_trt_pipeline()
{
    // 记录时间
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    std::string clip_engine_path = "/home/wangzijian/lite.ai.toolkit/examples/hub/trt/clip_model_fp16.engine";
    std::string unet_engine_path = "/home/wangzijian/lite.ai.toolkit/examples/hub/trt/unet_model_fp16.engine";
    std::string vae_engine_path = "/home/wangzijian/lite.ai.toolkit/examples/hub/trt/vae_model_fp16.engine";


    auto *pipeline = new lite::trt::sd::pipeline::PipeLine(
            clip_engine_path, unet_engine_path, vae_engine_path
    );


    std::string prompt = "1girl with red hair,blue eyes,smile, looking at viewer";
    std::string negative_prompt = "";
    std::string save_path = "/home/wangzijian/lite.ai.toolkit/output_merge_tensorrt.png";
    pipeline->inference(prompt,negative_prompt,save_path);

    // 记录结束时间并且输出
    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds" << std::endl;

    delete pipeline;

}

static void test_lite()
{
//    test_default();
    test_trt_pipeline();
}

int main()
{
    test_lite();
}