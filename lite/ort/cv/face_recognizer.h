//
// Created by wangzijian on 11/4/24.
//

#ifndef LITE_AI_TOOLKIT_FACE_RECOGNIZER_H
#define LITE_AI_TOOLKIT_FACE_RECOGNIZER_H
#include "lite/ort/core/ort_core.h"
#include "lite/ort/core/ort_types.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/ort/cv/face_utils.h"
namespace ortcv{
    class LITE_EXPORTS Face_Recognizer : public BasicOrtHandler{
    public:
        explicit  Face_Recognizer(const std::string &_onnx_path, unsigned int _num_threads = 1):
                BasicOrtHandler(_onnx_path, _num_threads = 1){};

        ~Face_Recognizer() override = default;

    private:
        cv::Mat  preprocess(cv::Mat &input_mat, std::vector<cv::Point2f> &face_landmark_5,cv::Mat &preprocessed_mat);

        Ort::Value transform(const cv::Mat &mat_rs) override;

    public:
        void detect(cv::Mat &input_mat,std::vector<cv::Point2f> &face_landmark_5);

        void detect(cv::Mat &input_mat,std::vector<cv::Point2f> &face_landmark_5,std::vector<float> &embeding);

    };
}


#endif //LITE_AI_TOOLKIT_FACE_RECOGNIZER_H
