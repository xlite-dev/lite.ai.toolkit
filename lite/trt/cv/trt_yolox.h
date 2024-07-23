//
// Created by wangzijian on 7/22/24.
//

#ifndef LITE_AI_TOOLKIT_TRT_YOLOX_H
#define LITE_AI_TOOLKIT_TRT_YOLOX_H
#include "lite/trt/core/trt_core.h"
#include "lite/utils.h"
#include "lite/trt/core/trt_utils.h"

namespace trtcv
{
    class LITE_EXPORTS TRTYoloX : public BasicTRTHandler
    {
    public:
        explicit TRTYoloX(const std::string &_trt_model_path, unsigned int _num_threads = 1) :
                BasicTRTHandler(_trt_model_path, _num_threads)
        {};

        ~TRTYoloX() override = default;

    private:
        // nested classes
        typedef struct GridAndStride
        {
            int grid0;
            int grid1;
            int stride;
        } YoloXAnchor;

        typedef struct
        {
            float r;
            int dw;
            int dh;
            int new_unpad_w;
            int new_unpad_h;
            bool flag;
        } YoloXScaleParams;


    private:
        const float mean_vals[3] = {255.f * 0.485f, 255.f * 0.456, 255.f * 0.406f};
        const float scale_vals[3] = {1 / (255.f * 0.229f), 1 / (255.f * 0.224f), 1 / (255.f * 0.225f)};

        const char *class_names[80] = {
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                "scissors", "teddy bear", "hair drier", "toothbrush"
        };
        enum NMS
        {
            HARD = 0, BLEND = 1, OFFSET = 2
        };
        static constexpr const unsigned int max_nms = 30000;


    private:
        void resize_unscale(const cv::Mat &mat,
                            cv::Mat &mat_rs,
                            int target_height,
                            int target_width,
                            YoloXScaleParams &scale_params);

        void generate_anchors(const int target_height,
                              const int target_width,
                              std::vector<int> &strides,
                              std::vector<YoloXAnchor> &anchors);

        void normalized(cv::Mat &mat_inplace, const float *mean, const float *scale);

        void generate_bboxes(const YoloXScaleParams &scale_params,
                             std::vector<types::Boxf> &bbox_collection,
                             float* output,
                             float score_threshold, int img_height,
                             int img_width); // rescale & exclude

        void nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                 float iou_threshold, unsigned int topk, unsigned int nms_type);

    public:
        void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                    float score_threshold = 0.25f, float iou_threshold = 0.45f,
                    unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);

    };

}

#endif //LITE_AI_TOOLKIT_TRT_YOLOX_H
