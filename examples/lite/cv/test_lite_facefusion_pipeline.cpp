//
// Created by wangzijian on 11/7/24.
//
#include "lite/lite.h"
static void test_default()
{
#ifdef ENABLE_ONNXRUNTIME
    std::string face_swap_onnx_path = "../../../examples/hub/onnx/cv/inswapper_128.onnx";
    std::string face_detect_onnx_path = "../../../examples/hub/onnx/cv/yoloface_8n.onnx";
    std::string face_landmarks_68 = "../../../examples/hub/onnx/cv/2dfan4.onnx";
    std::string face_recognizer_onnx_path = "../../../examples/hub/onnx/cv/arcface_w600k_r50.onnx";
    std::string face_restoration_onnx_path = "../../../examples/hub/onnx/cv/gfpgan_1.4.onnx";

    auto pipeLine =  lite::cv::face::swap::facefusion::PipeLine(
            face_detect_onnx_path,
            face_landmarks_68,
            face_recognizer_onnx_path,
            face_swap_onnx_path,
            face_restoration_onnx_path
            );

    std::string source_image_path = "../../../examples/lite/resources/test_lite_facefusion_pipeline_source.jpg";
    std::string target_image_path = "../../../examples/lite/resources/test_lite_facefusion_pipeline_target.jpg";
    std::string save_image_path = "../../../examples/logs/test_lite_facefusion_pipeline_result.jpg";


    // 写一个测试时间的代码
    auto start = std::chrono::high_resolution_clock::now();



    pipeLine.detect(source_image_path,target_image_path,save_image_path);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "Time: " << diff.count() << " s\n";


#endif
}

int main()
{

    test_default();
}