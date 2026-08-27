//
// Created by zhuohaoyang on 2026/8/26.
//

#ifndef LITE_AI_TOOLKIT_ORT_CV_YOLOV26_H
#define LITE_AI_TOOLKIT_ORT_CV_YOLOV26_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS YoloV26 : public BasicOrtHandler
  {
  public:
    explicit YoloV26(const std::string &_onnx_path, unsigned int _num_threads = 1);

    ~YoloV26() override = default;

  private:
    typedef struct
    {
      float scale;
      int left;
      int top;
    } ScaleParams;

  private:
    Ort::Value transform(const cv::Mat &mat_rs) override;

    void letterbox(const cv::Mat &mat, cv::Mat &mat_rs,
                   int target_height, int target_width,
                   ScaleParams &scale_params);

    void generate_bboxes(const ScaleParams &scale_params,
                         std::vector<types::Boxf> &detected_boxes,
                         std::vector<Ort::Value> &output_tensors,
                         float score_threshold, unsigned int topk,
                         int img_height, int img_width);

  public:
    void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                float score_threshold = 0.25f, unsigned int topk = 100);
  };
}

#endif //LITE_AI_TOOLKIT_ORT_CV_YOLOV26_H
