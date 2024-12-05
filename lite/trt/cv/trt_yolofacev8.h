//
// Created by wangzijian on 24-7-11.
//

#ifndef LITE_AI_TOOLKIT_TRT_YOLOFACEV8_H
#define LITE_AI_TOOLKIT_TRT_YOLOFACEV8_H
#include "lite/trt/core/trt_core.h"
#include "lite/trt/core/trt_utils.h"
#include "lite/trt/kernel/nms_cuda_manager.h"
#include "lite/trt/kernel/generate_bbox_cuda_manager.h"
#include "lite/trt/kernel/bgr2rgb.cuh"

namespace trtcv{
    class LITE_EXPORTS TRTYoloFaceV8 : public BasicTRTHandler{


    public:

        std::unique_ptr<NMSCudaManager> nms_cuda_manager;
        explicit TRTYoloFaceV8(const std::string& _trt_model_path,unsigned int _num_threads = 1):
                BasicTRTHandler(_trt_model_path, _num_threads)
        {
            nms_cuda_manager = std::make_unique<NMSCudaManager>();
        };



        std::vector<int> nms_cuda(std::vector<lite::types::Boxf> boxes,
                                  std::vector<float> confidences,
                                  const float nms_thresh);

    private:
        float mean = -127.5 / 128.0;
        float scale = 1 / 128.0;

        float ratio_width ;
        float ratio_height;

    private:
        // transform func

        float get_iou(const lite::types::Boxf box1, const lite::types::Boxf box2);

        std::vector<int> nms(std::vector<lite::types::Boxf> boxes, std::vector<float> confidences, const float nms_thresh);

        cv::Mat normalize(cv::Mat srcImg);

        void generate_box(float* trt_outputs, std::vector<lite::types::Boxf>& boxes,float conf_threshold, float iou_threshold);
    public:
        void detect(const cv::Mat &mat,std::vector<lite::types::Boxf> &boxes,
                    float conf_threshold, float iou_threshold);
    };
}


#endif //LITE_AI_TOOLKIT_TRT_YOLOFACEV8_H
