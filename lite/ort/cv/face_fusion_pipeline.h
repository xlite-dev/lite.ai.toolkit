//
// Created by wangzijian on 11/7/24.
//

#ifndef LITE_AI_TOOLKIT_FACE_FUSION_PIPELINE_H
#define LITE_AI_TOOLKIT_FACE_FUSION_PIPELINE_H
#include "lite/ort/core/ort_core.h"
#include "lite/ort/cv/face_restoration.h"
#include "lite/ort/cv/face_swap.h"
#include "lite/ort/cv/face_recognizer.h"
#include "lite/ort/cv/yolofacev8.h"
#include "lite/ort/cv/face_68landmarks.h"

namespace ortcv{
    class Face_Fusion_Pipeline{
    public:
        Face_Fusion_Pipeline(
                const std::string &face_detect_onnx_path,
                const std::string &face_landmarks_68_onnx_path,
                const std::string &face_recognizer_onnx_path,
                const std::string &face_swap_onnx_path,
                const std::string &face_restoration_onnx_path
                );
        ~Face_Fusion_Pipeline() = default; // 使用智能指针来进行管理

    private:
        std::unique_ptr<Face_Restoration> face_restoration;
        std::unique_ptr<YoloFaceV8> face_detect;
        std::unique_ptr<Face_68Landmarks> face_landmarks;
        std::unique_ptr<Face_Recognizer> face_recognizer;
        std::unique_ptr<Face_Swap> face_swap;

    public:
        void detect(const std::string &source_image,const std::string &target_image,const std::string &save_image);
    };
}

#endif //LITE_AI_TOOLKIT_FACE_FUSION_PIPELINE_H
