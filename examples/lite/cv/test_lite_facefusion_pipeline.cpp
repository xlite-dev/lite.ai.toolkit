//
// Created by wangzijian on 11/7/24.
//
#include "lite/lite.h"
static void test_default()
{
#ifdef ENABLE_ONNXRUNTIME
    std::string face_swap_onnx_path = "/home/lite.ai.toolkit/examples/hub/onnx/cv/inswapper_128.onnx";
    std::string face_detect_onnx_path = "/home/lite.ai.toolkit/examples/hub/onnx/cv/yoloface_8n.onnx";
    std::string face_landmarks_68 = "/home/lite.ai.toolkit/examples/hub/onnx/cv/2dfan4.onnx";
    std::string face_recognizer_onnx_path = "/home/lite.ai.toolkit/examples/hub/onnx/cv/arcface_w600k_r50.onnx";
    std::string face_restoration_onnx_path = "/home/lite.ai.toolkit/examples/hub/onnx/cv/gfpgan_1.4.onnx";

    auto pipeLine =  lite::cv::face::swap::facefusion::PipeLine(
            face_detect_onnx_path,
            face_landmarks_68,
            face_recognizer_onnx_path,
            face_swap_onnx_path,
            face_restoration_onnx_path
            );

    std::string source_image_path = "/home/lite.ai.toolkit/1.jpg";
    std::string target_image_path = "/home/lite.ai.toolkit/2.jpg";
    std::string save_image_path = "/home/lite.ai.toolkit/result111111.jpg";


    // 写一个测试时间的代码
    auto start = std::chrono::high_resolution_clock::now();

    pipeLine.detect(source_image_path,target_image_path,save_image_path);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "Time: " << diff.count() << " s\n";


#endif
}




static void test_tensorrt()
{
#ifdef ENABLE_TENSORRT
    std::string face_swap_onnx_path = "../../../examples/hub/trt/inswapper_128_fp16.engine";
    std::string face_detect_onnx_path = "../../../examples/hub/trt/yoloface_8n_fp16.engine";
    std::string face_landmarks_68 = "../../../examples/hub/trt/2dfan4_fp16.engine";
    std::string face_recognizer_onnx_path = "../../../examples/hub/trt/arcface_w600k_r50_fp16.engine";
    std::string face_restoration_onnx_path = "../../../examples/hub/trt/gfpgan_1.4_fp32.engine";

    auto pipeLine =  lite::trt::cv::face::swap::FaceFusionPipeLine (
            face_detect_onnx_path,
            face_landmarks_68,
            face_recognizer_onnx_path,
            face_swap_onnx_path,
            face_restoration_onnx_path
    );

    std::string source_image_path = "../../../examples/logs/1.jpg";
    std::string target_image_path = "../../../examples/logs/5.jpg";
    std::string save_image_path = "../../../examples/logs/trt_pipeline_result_cuda_test_13_mt.jpg";


    // 写一个测试时间的代码
    auto start = std::chrono::high_resolution_clock::now();

    pipeLine.detect(source_image_path,target_image_path,save_image_path);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "Time: " << diff.count()  * 1000<< " ms\n";


#endif
}

int main()
{
    test_tensorrt();
//    test_default();
}