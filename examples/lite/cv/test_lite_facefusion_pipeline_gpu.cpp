//
// Created by wangzijian on 3/5/25.
//
#include "lite/trt/cv/trt_facefusion_pipeline_gpu.h"

void test_default(){
    std::string face_swap_onnx_path = "/home/lite.ai.toolkit/examples/hub/trt/inswapper_128_fp16.engine";
    std::string face_detect_onnx_path = "/home/lite.ai.toolkit/examples/hub/trt/yoloface_8n_fp16.engine";
    std::string face_landmarks_68 = "/home/lite.ai.toolkit/examples/hub/trt/2dfan4_fp16.engine";
    std::string face_recognizer_onnx_path = "/home/lite.ai.toolkit/examples/hub/trt/arcface_w600k_r50_fp16.engine";
    std::string face_restoration_onnx_path = "/home/lite.ai.toolkit/examples/hub/trt/gfpgan_1.4_fp32.engine";
    std::vector<std::string> model_list{face_swap_onnx_path,face_detect_onnx_path,face_landmarks_68,
                                        face_recognizer_onnx_path,face_restoration_onnx_path};

    trt_facefusion_pipeline_gpu test(model_list);
    cv::Mat test1 = cv::imread("/home/lite.ai.toolkit/1.jpg");
    cv::Mat test2;

    test.detect(test1,test2,1,2);
}


int main(){
    test_default();
}