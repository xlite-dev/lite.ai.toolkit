//
// Created by wangzijian on 11/7/24.
//

#ifndef LITE_AI_TOOLKIT_FACE_RESTORATION_H
#define LITE_AI_TOOLKIT_FACE_RESTORATION_H
#include "lite/ort/core/ort_core.h"
#include "lite/ort/core/ort_types.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/ort/cv/face_utils.h"

namespace ortcv{
    class LITE_EXPORTS Face_Restoration : public BasicOrtHandler{
    public:
        explicit Face_Restoration(const std::string &_onnx_path, unsigned int _num_threads = 1):
                BasicOrtHandler(_onnx_path,_num_threads){};
        ~Face_Restoration() override = default;

    private:

        Ort::Value transform(const cv::Mat &mat_rs) override;

    public:
        void detect(cv::Mat &face_swap_image,std::vector<cv::Point2f > &target_landmarks_5 ,const std::string &face_enchaner_path);
    };
}

#endif //LITE_AI_TOOLKIT_FACE_RESTORATION_H
