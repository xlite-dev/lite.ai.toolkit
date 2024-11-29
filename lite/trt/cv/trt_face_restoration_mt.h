// trt_face_restoration_mt.h
#ifndef LITE_AI_TOOLKIT_TRT_FACE_RESTORATION_MT_H_  // 注意添加_MT后缀
#define LITE_AI_TOOLKIT_TRT_FACE_RESTORATION_MT_H_

#include "cuda_runtime.h"
#include "NvInfer.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "lite/trt/core/trt_logger.h"
#include "lite/ort/cv/face_utils.h"
#include "lite/trt/core/trt_utils.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <memory>
#include <fstream>


// 定义任务结构体
struct InferenceTaskTest {
    cv::Mat face_swap_image;
    std::vector<cv::Point2f> target_landmarks_5;
    std::string face_enchaner_path;
};

class trt_face_restoration_mt {
private:
    Logger logger;

    // TensorRT相关组件
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
    std::queue<InferenceTaskTest> task_queue;
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
    void process_single_task(const InferenceTaskTest& task, int thread_id);

public:
    explicit trt_face_restoration_mt(std::string& model_path, size_t num_threads = 4);
    ~trt_face_restoration_mt();

    // 异步任务提交接口
    void detect_async(cv::Mat& face_swap_image,
                      std::vector<cv::Point2f>& target_landmarks_5,
                      const std::string& face_enchaner_path);

    void shutdown(); // 新增：显式关闭方法

    // 等待所有任务完成
    void wait_for_completion();
};
#endif