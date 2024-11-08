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
    std::string target_image_path = "/home/lite.ai.toolkit/5.jpg";
    std::string save_image_path = "/home/lite.ai.toolkit/final_pipeline.jpg";

    pipeLine.inference(source_image_path,target_image_path,save_image_path);


#endif
}

int main()
{
    test_default();
}