//
// Created by root on 11/15/24.
//

#ifndef LITE_AI_TOOLKIT_TRT_FACE_RESTORATION_MT_H
#define LITE_AI_TOOLKIT_TRT_FACE_RESTORATION_MT_H
#include "cuda_runtime.h"
#include "NvInfer.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "lite/trt/core/trt_logger.h"
#include "lite/ort/cv/face_utils.h"
#include "lite/trt/core/trt_utils.h"

class trt_face_restoration_mt {
private:

    Logger logger;

    std::unique_ptr<nvinfer1::IRuntime> trt_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> trt_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> trt_context;
    std::vector<void*> buffers;
    cudaStream_t stream;
    std::vector<int64_t> input_node_dims;
    std::vector<std::vector<int64_t>> output_node_dims;
    std::size_t input_tensor_size = 1;
    std::size_t output_tensor_size = 0;

public:
    void detect(cv::Mat &face_swap_image,std::vector<cv::Point2f > &target_landmarks_5 ,const std::string &face_enchaner_path);

    trt_face_restoration_mt(std::string &model_path);

    ~trt_face_restoration_mt();

};


#endif //LITE_AI_TOOLKIT_TRT_FACE_RESTORATION_MT_H

