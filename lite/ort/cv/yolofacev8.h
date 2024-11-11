//
// Created by ai-test1 on 24-7-8.
//

#ifndef LITE_AI_TOOLKIT_YOLOFACEV8_H
#define LITE_AI_TOOLKIT_YOLOFACEV8_H
#include "lite/ort/core/ort_core.h"
#include "algorithm"
#include <vector>

namespace ortcv {

    class LITE_EXPORTS YoloFaceV8 : public BasicOrtHandler{
    public:
        explicit YoloFaceV8(const std::string &_onnx_path, unsigned int _num_threads = 1) :
                BasicOrtHandler(_onnx_path, _num_threads)
        {};
        ~YoloFaceV8() override = default;

    private:
        float mean = -127.5 / 128.0;
        float scale = 1 / 128.0;
        float ratio_width;
        float ratio_height;


    private:
        // need override's function
        Ort::Value transform(const cv::Mat &mat_rs) override;


        float get_iou(const lite::types::Boxf box1, const lite::types::Boxf box2);

        std::vector<int> nms(std::vector<lite::types::Boxf> boxes, std::vector<float> confidences, const float nms_thresh);

        cv::Mat normalize(cv::Mat srcImg);

        void generate_box(std::vector<Ort::Value> &ort_outputs, std::vector<lite::types::Boxf> &boxes,
                          float conf_threshold = 0.25f, float iou_threshold = 0.45f);


    public:

        void detect(const cv::Mat &mat, std::vector<lite::types::Boxf> &boxes, 
                    float conf_threshold = 0.25f, float iou_threshold = 0.45f);
    };
}


#endif //LITE_AI_TOOLKIT_YOLOFACEV8_H
