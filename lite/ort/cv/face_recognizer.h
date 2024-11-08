//
// Created by wangzijian on 11/4/24.
//

#ifndef LITE_AI_TOOLKIT_FACE_RECOGNIZER_H
#define LITE_AI_TOOLKIT_FACE_RECOGNIZER_H
#include "lite/ort/core/ort_core.h"
#include "lite/ort/core/ort_types.h"
#include "lite/ort/core/ort_utils.h"
namespace ortcv{
    class LITE_EXPORTS Face_Recognizer : public BasicOrtHandler{
    public:
        explicit  Face_Recognizer(const std::string &_onnx_path, unsigned int _num_threads = 1):
                BasicOrtHandler(_onnx_path, _num_threads = 1){};

        ~Face_Recognizer() override = default;

    private:
        cv::Mat  preprocess(cv::Mat &input_mat, std::vector<cv::Point2f> &face_landmark_5,cv::Mat &preprocessed_mat);

        Ort::Value transform(const cv::Mat &mat_rs) override;

        std::pair<cv::Mat, cv::Mat> warp_face_by_face_landmark_5(cv::Mat input_mat,std::vector<cv::Point2f> face_landmark_5);

    private:
        std::vector<cv::Point2f> face_template = {
                cv::Point2f(0.34191607, 0.46157411),
                cv::Point2f(0.65653393, 0.45983393),
                cv::Point2f(0.50022500, 0.64050536),
                cv::Point2f(0.37097589, 0.82469196),
                cv::Point2f(0.63151696, 0.82325089)
        };


    public:
        void detect(cv::Mat &input_mat,std::vector<cv::Point2f> &face_landmark_5);

        void detect(cv::Mat &input_mat,std::vector<cv::Point2f> &face_landmark_5,std::vector<float> &embeding);

    };
}


#endif //LITE_AI_TOOLKIT_FACE_RECOGNIZER_H
