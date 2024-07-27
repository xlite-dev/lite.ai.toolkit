//
// Created by wangzijian on 7/27/24.
//

#ifndef LITE_AI_TOOLKIT_YOLOV5_BLAZEFACE_H
#define LITE_AI_TOOLKIT_YOLOV5_BLAZEFACE_H


#include "lite/trt/core/trt_core.h"
#include "lite/trt/core/trt_utils.h"


namespace trtcv{
    class LITE_EXPORTS TRTYOLO5Face : public BasicTRTHandler{
    public:
        explicit TRTYOLO5Face(const std::string& _trt_model_path,unsigned int _num_threads = 1):
                BasicTRTHandler(_trt_model_path, _num_threads)
        {};
        ~TRTYOLO5Face() override = default;


    private:
        // nested classes
        typedef struct
        {
            float ratio;
            int dw;
            int dh;
            bool flag;
        } YOLOv5BlazeFaceScaleParams;

    private:
        static constexpr const float mean_val = 0.f; // RGB
        static constexpr const float scale_val = 1.0 / 255.f;
        static constexpr const unsigned int max_nms = 30000;

    private:
        // transform func

        void resize_unscale(const cv::Mat &mat,
                            cv::Mat &mat_rs,
                            int target_height,
                            int target_width,
                            YOLOv5BlazeFaceScaleParams &scale_params);

        void generate_bboxes_kps(const YOLOv5BlazeFaceScaleParams &scale_params,
                                 std::vector<types::BoxfWithLandmarks> &bbox_kps_collection,
                                 float* trt_outputs,
                                 float score_threshold, float img_height,
                                 float img_width); // rescale & exclude

        void normalized(cv::Mat &input_image);

        void nms_bboxes_kps(std::vector<types::BoxfWithLandmarks> &input,
                            std::vector<types::BoxfWithLandmarks> &output,
                            float iou_threshold, unsigned int topk);

    public:
        void detect(const cv::Mat &mat, std::vector<types::BoxfWithLandmarks> &detected_boxes_kps,
                    float score_threshold = 0.3f, float iou_threshold = 0.45f,
                    unsigned int topk = 400);
    };
}
#endif //LITE_AI_TOOLKIT_YOLOV5_BLAZEFACE_H
