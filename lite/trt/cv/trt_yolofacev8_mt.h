//
// Created by root on 12/3/24.
//

#ifndef LITE_AI_TOOLKIT_TRT_YOLOFACEV8_MT_H
#define LITE_AI_TOOLKIT_TRT_YOLOFACEV8_MT_H
#include "cuda_runtime.h"
#include "NvInfer.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "lite/trt/core/trt_logger.h"
#include "lite/ort/cv/face_utils.h"
#include "lite/trt/core/trt_utils.h"
#include "lite/types.h"
#include "queue"
#include "mutex"
#include "condition_variable"
#include "thread"
#include "atomic"
#include "memory"
#include "fstream"
#include "future"
#include "lite/trt/kernel/nms_cuda_manager.h"
#include "lite/trt/kernel/generate_bbox_cuda_manager.h"

// 需要定义任务的结构体
struct InferenceTaskFace{
    cv::Mat input_mat; // 送入的检测的mat
    std::vector<lite::types::Boxf>* bbox; // 得到的结果
    std::promise<void> completion_promise;
};

class trt_yolofacev8_mt{
public:
    Logger logger;
    // 多线程需要给每个线程进行分配资源
    std::unique_ptr<nvinfer1::IRuntime> trt_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> trt_engine;
    std::vector<std::unique_ptr<nvinfer1::IExecutionContext>> trt_contexts;  // 每个线程一个context
    std::vector<cudaStream_t> streams;  // 每个线程一个stream
    std::vector<std::vector<void*>> buffers;  // 每个线程一组buffer

    // 模型相关维度信息
    std::vector<int64_t> input_node_dims;
    std::vector<std::vector<int64_t>> output_node_dims;
    std::size_t input_tensor_size = 1;
    std::size_t output_tensor_size = 0;

    // 线程池相关组件
    std::vector<std::thread> worker_threads;
    std::queue<InferenceTaskFace> task_queue;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop_flag{false};
    size_t num_threads;

    std::atomic<int> active_tasks{0};  // 新增：跟踪活跃任务数
    std::mutex completion_mutex;
    std::condition_variable completion_cv;

    // 线程工作函数
    void worker_function(int thread_id);

    // 实际的推理函数
    void process_single_task(InferenceTaskFace& task, int thread_id);

    cv::Mat normalize(cv::Mat srcImg,int thread_id);

    void generate_box(float* trt_outputs, std::vector<lite::types::Boxf>& boxes,float conf_threshold, float iou_threshold,int thread_id);

    float get_iou(const lite::types::Boxf box1, const lite::types::Boxf box2);

    float mean = -127.5 / 128.0;
    float scale = 1 / 128.0;

    std::vector<float> ratio_width;
    std::vector<float> ratio_height;

    std::unique_ptr<NMSCudaManager> nms_cuda_manager = std::make_unique<NMSCudaManager>();

    std::vector<int> nms_cuda(std::vector<lite::types::Boxf> boxes,
                              std::vector<float> confidences,
                              const float nms_thresh);


public:
    explicit trt_yolofacev8_mt(std::string& model_path, size_t num_threads = 4);
    ~trt_yolofacev8_mt();

    // 异步任务提交接口
//    void detect_async(cv::Mat& input_image, const lite::types::Boxf& bbox, std::vector<cv::Point2f>& face_landmark_5of68);
    void detect_async(cv::Mat &input_image, std::vector<lite::types::Boxf>* bbox);
    void shutdown(); // 新增：显式关闭方法

    // 等待所有任务完成
    void wait_for_completion();
};


#endif //LITE_AI_TOOLKIT_TRT_YOLOFACEV8_MT_H
