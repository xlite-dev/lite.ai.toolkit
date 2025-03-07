//
// Created by wangzijian on 3/5/25.
//

#ifndef LITE_AI_TOOLKIT_TRT_FACEFUSION_PIPELINE_GPU_H
#define LITE_AI_TOOLKIT_TRT_FACEFUSION_PIPELINE_GPU_H

#include "cuda_runtime.h"
#include "NvInferRuntime.h"
#include "NvInfer.h"
#include "fstream"
#include "opencv2/opencv.hpp"
#include "iostream"
#include "lite/types.h"
#include "lite/utils.h"
#include <algorithm>
using namespace nvinfer1;

#include <numeric>    // std::accumulate
#include <functional> // std::multiplies

inline int volume(const nvinfer1::Dims &dims) {
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
}

// 定义错误宏
#define CHECK_CUDA(call) {                                    \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
}



// Logger for TensorRT info/warning/errors
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // suppress info-level messages
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
};



class trt_facefusion_pipeline_gpu {
    // 构造函数获取四个model的path
public:
    void detect(cv::Mat &src_image,cv::Mat &target_image,int idx_of_src,int idx_of_target);

    void detect_landmarks_68(lite::types::Boxf* d_output_boxes,int* d_number_of_boxes);

    trt_facefusion_pipeline_gpu(const std::vector<std::string> &model_path_list);

    ~trt_facefusion_pipeline_gpu();

private:
    Logger trt_logger;

    std::unique_ptr<nvinfer1::IRuntime> trt_runtime;// 全局唯一的 runtime
    std::vector<std::unique_ptr<ICudaEngine>> engines;  // 四个模型的引擎
    std::vector<std::unique_ptr<nvinfer1::IExecutionContext>> contexts; // 四个模型的执行上下文

    // 2. GPU 内存管理
    struct ModelIO{
        void* d_input;      // 输入数据 GPU 指针
        void* d_output;     // 输出数据 GPU 指针
        size_t input_size;  // 输入数据字节数
        size_t output_size; // 输出数据字节数
    };
    std::vector<ModelIO> model_ios; // 每个模型的输入输出内存

    // 3. 中间结果缓存（例如人脸框、关键点、换脸结果等）
    // TODO可能会有新增
    void* d_bboxes;    // yolov8face 输出的人脸框
    void* d_landmarks; // facelandmarks 输出的关键点
    void* d_swapped;   // inswaper 换脸结果
    void* d_final;     // gfpgan 修复后的最终输出

    // 4. CUDA Stream 和事件（可选）
    // 目前是单帧所以一个stream就可以完成
    cudaStream_t stream; // 全局 Stream（简单情况）或多个 Stream（流水线优化）

    //5. 记录输入和输出维度 防止需要
    std::vector<std::vector<int>> input_dims_list;
    std::vector<std::vector<int>> output_dims_list;

    float ratio_height;
    float ratio_width;


    // 初始化模型引擎和内存
    void init_engine(const std::string& model_path, int model_idx);

    void normalize_cuda_direct(cv::Mat srcimg, float* output_gpu_buffer);

    // Main wrapper function that handles the entire post-processing pipeline
    void process_yolov8_detections_gpu(
            float* face_detect_infer,
            int num_boxes,
            float conf_threshold,
            float iou_threshold,
            float ratio_height,
            float ratio_width,
            std::vector<lite::types::Boxf>& output_boxes,
            cudaStream_t stream,
            lite::types::Boxf** d_output_boxes_ptr,
            int* output_box_count
    );
};


#endif //LITE_AI_TOOLKIT_TRT_FACEFUSION_PIPELINE_GPU_H
