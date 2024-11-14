//
// Created by wangzijian on 11/14/24.
//

#ifndef LITE_AI_TOOLKIT_TRT_FACEFUSION_PIPELINE_H
#define LITE_AI_TOOLKIT_TRT_FACEFUSION_PIPELINE_H

#include "lite/trt/core/trt_core.h"
#include "lite/trt/cv/trt_face_restoration.h"
#include "lite/trt/cv/trt_face_swap.h"
#include "lite/trt/cv/trt_face_recognizer.h"
#include "lite/trt/cv/trt_yolofacev8.h"
#include "lite/trt/cv/trt_face_68landmarks.h"

namespace trtcv{
    class TRTFaceFusionPipeLine{
    public:
        TRTFaceFusionPipeLine(
                const std::string &face_detect_engine_path,
                const std::string &face_landmarks_68_engine_path,
                const std::string &face_recognizer_engine_path,
                const std::string &face_swap_engine_path,
                const std::string &face_restoration_engine_path
                );

    private:
        std::unique_ptr<TRTFaceFusionFaceRestoration> face_restoration;
        std::unique_ptr<TRTYoloFaceV8> face_detect;
        std::unique_ptr<TRTFaceFusionFace68Landmarks> face_landmarks;
        std::unique_ptr<TRTFaceFusionFaceRecognizer> face_recognizer;
        std::unique_ptr<TRTFaceFusionFaceSwap> face_swap;

    public:
        void detect(const std::string &source_image,const std::string &target_image,const std::string &save_image);

    };
}


#endif //LITE_AI_TOOLKIT_TRT_FACEFUSION_PIPELINE_H
