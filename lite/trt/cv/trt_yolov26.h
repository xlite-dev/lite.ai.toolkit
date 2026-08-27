//
// Created by zhuohaoyang on 2026/8/27.
//

#ifndef LITE_AI_TOOLKIT_TRT_YOLOV26_H
#define LITE_AI_TOOLKIT_TRT_YOLOV26_H

#include "lite/trt/core/trt_core.h"

namespace trtcv
{
  class LITE_EXPORTS TRTYoloV26 : public BasicTRTHandler
  {
  public:
    explicit TRTYoloV26(const std::string &_trt_model_path,
                       unsigned int _num_threads = 1);

    ~TRTYoloV26() override = default;

  private:
    typedef struct
    {
      float scale;
      int left;
      int top;
    } ScaleParams;

  private:
    void letterbox(const cv::Mat &mat, cv::Mat &mat_rs,
                   int target_height, int target_width,
                   ScaleParams &scale_params);

    void generate_bboxes(const ScaleParams &scale_params,
                         std::vector<types::Boxf> &detected_boxes,
                         const float *output, std::size_t num_predictions,
                         float score_threshold, unsigned int topk,
                         int img_height, int img_width);

  public:
    void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                float score_threshold = 0.25f, unsigned int topk = 100);
  };
}

#endif //LITE_AI_TOOLKIT_TRT_YOLOV26_H
