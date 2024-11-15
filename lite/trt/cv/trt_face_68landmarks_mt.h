//
// Created by root on 11/15/24.
//

#ifndef LITE_AI_TOOLKIT_TRT_FACE_68LANDMARKS_MT_H
#define LITE_AI_TOOLKIT_TRT_FACE_68LANDMARKS_MT_H
#include "cuda_runtime.h"
#include "NvInfer.h"
#include "opencv2/core.hpp"
#include "lite/trt/core/trt_logger.h"
#include "lite/ort/cv/face_utils.h"
#include "lite/ort/core/ort_types.h"
#include "lite/trt/core/trt_utils.h"

// 多线程需要的一些头文件
#include "queue"
#include "mutex"
#include "condition_variable"
#include "thread"
#include "atomic"
#include "memory"

// 定义任务结构体
struct InferenceTask{
    cv::Mat input_mat; // 输入
    lite::types::BoundingBoxType<float, float> bbox; //输入
    std::vector<cv::Point2f> face_landmark_5of68; // 输出
};



class trt_face_68landmarks_mt {
private:
    Logger logger;

    // TensorRT 的不需要线程安全的变量 也就是可以同时访问的
    std::unique_ptr<nvinfer1::IRuntime> trt_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> trt_engine;
    // 这里不能只初始化一个context，因为多线程的时候，每个线程都需要一个context
    std::unique_ptr<std::unique_ptr<nvinfer1::IExecutionContext>> trt_context;
    // 每组一个 buffer
    std::unique_ptr<std::unique_ptr<void>> buffers;
    // 每组一个 stream
    std::unique_ptr<std::unique_ptr<cudaStream_t>> stream;

    // 模型维度是固定的每个线程都可以共享
    std::vector<int64_t> input_node_dims;
    std::vector<std::vector<int64_t>> output_node_dims;
    // 这些输入和输出都是可以共享的 不需要线程安全
    std::size_t input_tensor_size = 1;
    std::size_t output_tensor_size = 0;

    // 线程池的相关组件
    std::vector<std::thread> threads; // 线程池 = 工厂中的工人
    std::queue<InferenceTask> task_queue; // 任务队列 = 任务堆
    std::mutex task_queue_mutex; // 任务队列锁 = 任务堆锁 防止多线程同时操作任务队列
    std::condition_variable task_queue_cv; // 唤醒队列中的线程 = 唤醒工厂中的工人
    std::atomic<bool> stop_flag = false; // 停止标志 = 工厂是否关闭
    size_t num_threads; // 线程数量 = 工人数量

    std::atomic<int> active_tasks{0};  // 新增：跟踪活跃任务数
    std::mutex completion_mutex; // 新增：完成队列锁
    std::condition_variable completion_cv; // 新增：完成所有任务之后更新

    // 线程工作函数
    void worker_thread(int thread_id);

    // 实际的单次推理函数
    void process_single_task(InferenceTask& task, int thread_id);

public:
    explicit trt_face_68landmarks_mt(std::string& model_path, size_t num_threads =4);

    ~trt_face_68landmarks_mt();

    // 异步任务提交接口
    void detect_async( cv::Mat input_mat,lite::types::BoundingBoxType<float, float> bbox,
            std::vector<cv::Point2f> face_landmark_5of68);

    // 关闭线程池
    void shutdown();

    // 等待所有线程完成
    void wait_for_completion();



};


#endif //LITE_AI_TOOLKIT_TRT_FACE_68LANDMARKS_MT_H
