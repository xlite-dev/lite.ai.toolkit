//
// Created by root on 11/15/24.
//

#include "trt_face_68landmarks_mt.h"


trt_face_68landmarks_mt::trt_face_68landmarks_mt(std::string &model_path, size_t num_threads) : num_threads(num_threads){

    // 1. 读取模型文件
    std::ifstream file(model_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to read model file: " << model_path << std::endl;
        return;
    }

    file.seekg(0, std::ifstream::end);
    size_t model_size = file.tellg();
    file.seekg(0, std::ifstream::beg);
    std::vector<char> model_data(model_size);
    file.read(model_data.data(), model_size);
    file.close();

    // 2. 创建TensorRT运行时和引擎
    trt_runtime.reset(nvinfer1::createInferRuntime(logger));
    trt_engine.reset(trt_runtime->deserializeCudaEngine(model_data.data(), model_size));

    if (!trt_engine) {
        std::cerr << "Failed to deserialize the TensorRT engine." << std::endl;
        return;
    }

    // 3. 获取模型输入输出信息
    int num_io_tensors = trt_engine->getNbIOTensors();

    // 4. 为每个线程创建执行上下文和CUDA流
    trt_contexts.resize(num_threads);
    streams.resize(num_threads);
    buffers.resize(num_threads);

    for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
        // 创建执行上下文
        trt_contexts[thread_id].reset(trt_engine->createExecutionContext());
        if (!trt_contexts[thread_id]) {
            std::cerr << "Failed to create execution context for thread " << thread_id << std::endl;
            return;
        }

        // 创建CUDA流
        cudaStreamCreate(&streams[thread_id]);

        // 为每个线程分配输入输出缓冲区
        buffers[thread_id].resize(num_io_tensors);

        for (int i = 0; i < num_io_tensors; ++i) {
            auto tensor_name = trt_engine->getIOTensorName(i);
            nvinfer1::Dims tensor_dims = trt_engine->getTensorShape(tensor_name);

            // 处理输入tensor
            if (i == 0) {
                size_t tensor_size = 1;
                for (int j = 0; j < tensor_dims.nbDims; ++j) {
                    tensor_size *= tensor_dims.d[j];
                    if (thread_id == 0) {  // 只在第一个线程记录输入维度
                        input_node_dims.push_back(tensor_dims.d[j]);
                    }
                }
                cudaMalloc(&buffers[thread_id][i], tensor_size * sizeof(float));
                trt_contexts[thread_id]->setTensorAddress(tensor_name, buffers[thread_id][i]);
                continue;
            }

            // 处理输出tensor
            size_t tensor_size = 1;
            if (thread_id == 0) {  // 只在第一个线程记录输出维度
                std::vector<int64_t> output_node;
                for (int j = 0; j < tensor_dims.nbDims; ++j) {
                    output_node.push_back(tensor_dims.d[j]);
                    tensor_size *= tensor_dims.d[j];
                }
                output_node_dims.push_back(output_node);
            } else {
                for (int j = 0; j < tensor_dims.nbDims; ++j) {
                    tensor_size *= tensor_dims.d[j];
                }
            }

            cudaMalloc(&buffers[thread_id][i], tensor_size * sizeof(float));
            trt_contexts[thread_id]->setTensorAddress(tensor_name, buffers[thread_id][i]);

            if (thread_id == 0) {
                output_tensor_size++;
            }
        }
    }

    // 5. 启动工作线程
    for (size_t i = 0; i < num_threads; ++i) {
        worker_threads.emplace_back(&trt_face_68landmarks_mt::worker_function, this, i);
    }

    affine_matrixs.resize(num_threads);
    img_with_landmarks_vec.resize(num_threads);

}

// 在cpp文件中修改相关实现
void trt_face_68landmarks_mt::worker_function(int thread_id) {
    while (true) {
        InferenceTask task;
        bool has_task = false;

        // 从任务队列获取任务
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (!task_queue.empty()) {
                task = std::move(task_queue.front());
                task_queue.pop();
                has_task = true;
                active_tasks++;
            } else if (stop_flag) {
                break;
            } else {
                condition.wait(lock);
                continue;
            }
        }

        if (has_task) {
            // 处理任务
            process_single_task(task, thread_id);

            // 更新活跃任务计数
            {
                std::lock_guard<std::mutex> lock(completion_mutex);
                active_tasks--;
                completion_cv.notify_all();
            }
        }
    }
}

void
trt_face_68landmarks_mt::preprocess(const lite::types::Boxf &bounding_box, const cv::Mat &input_mat, cv::Mat &crop_img,int thread_id) {
    float xmin = bounding_box.x1;
    float ymin = bounding_box.y1;
    float xmax = bounding_box.x2;
    float ymax = bounding_box.y2;


    float width = xmax - xmin;
    float height = ymax - ymin;
    float max_side = std::max(width, height);
    float scale = 195.0f / max_side;

    float center_x = (xmax + xmin) * scale;
    float center_y = (ymax + ymin) * scale;

    cv::Point2f translation;
    translation.x = (256.0f - center_x) * 0.5f;
    translation.y = (256.0f - center_y) * 0.5f;

    cv::Size crop_size(256, 256);

    std::tie(crop_img, affine_matrixs[thread_id]) = face_utils::warp_face_by_translation(input_mat, translation, scale, crop_size);

    crop_img.convertTo(crop_img,CV_32FC3,1 / 255.f);
}


void trt_face_68landmarks_mt::process_single_task( InferenceTask &task, int thread_id) {
    if (task.input_mat.empty()) return;

    img_with_landmarks_vec[thread_id] = task.input_mat.clone();
    cv::Mat crop_image;

    preprocess(task.bbox, task.input_mat, crop_image, thread_id);

    std::vector<float> input_data;

    trtcv::utils::transform::create_tensor(crop_image,input_data,input_node_dims,trtcv::utils::transform::CHW);

    cudaMemcpyAsync(buffers[thread_id][0], input_data.data(), input_node_dims[0] * input_node_dims[1] * input_node_dims[2] * input_node_dims[3] * sizeof(float),
                    cudaMemcpyHostToDevice, streams[thread_id]);

    // 在推理之前同步流，確保數據完全拷貝
    cudaStreamSynchronize(streams[thread_id]);
    bool status = trt_contexts[thread_id]->enqueueV3(streams[thread_id]);
    cudaStreamSynchronize(streams[thread_id]);

    if (!status){
        std::cerr << "Failed to infer by TensorRT." << std::endl;
        return;
    }

    std::vector<float> output(output_node_dims[0][0] * output_node_dims[0][1] * output_node_dims[0][2]);
    cudaMemcpyAsync(output.data(), buffers[thread_id][1], output_node_dims[0][0] * output_node_dims[0][1] * output_node_dims[0][2] * sizeof(float),
                    cudaMemcpyDeviceToHost, streams[thread_id]);
    cudaStreamSynchronize(streams[thread_id]);


    // 带出结果
    // 指针指向带出来
    *task.face_landmark_5of68 = postprocess(output.data(),thread_id);

    task.completion_promise.set_value();
}


std::vector<cv::Point2f> trt_face_68landmarks_mt::postprocess(float *trt_outputs,int thread_id) {
    std::vector<cv::Point2f> landmarks;

    for (int i = 0;i < 68; ++i)
    {
        float x = trt_outputs[i * 3] / 64.0f  * 256.f;
        float y = trt_outputs[i * 3 + 1] / 64.0f * 256.f;
        landmarks.emplace_back(x, y);
    }

    cv::Mat inverse_affine_matrix;
    cv::invertAffineTransform(affine_matrixs[thread_id], inverse_affine_matrix);

    cv::transform(landmarks, landmarks, inverse_affine_matrix);

    return face_utils::convert_face_landmark_68_to_5(landmarks);
}


void trt_face_68landmarks_mt::postprocess(float *trt_outputs, std::vector<cv::Point2f> &face_landmark_5of68,int thread_id) {
    std::vector<cv::Point2f> landmarks;

    for (int i = 0;i < 68; ++i)
    {
        float x = trt_outputs[i * 3] / 64.0f  * 256.f;
        float y = trt_outputs[i * 3 + 1] / 64.0f * 256.f;
        landmarks.emplace_back(x, y);
    }

    cv::Mat inverse_affine_matrix;
    cv::invertAffineTransform(affine_matrixs[thread_id], inverse_affine_matrix);

    cv::transform(landmarks, landmarks, inverse_affine_matrix);

    face_landmark_5of68 = face_utils::convert_face_landmark_68_to_5(landmarks);
}

void trt_face_68landmarks_mt::detect_async(cv::Mat &input_image, const lite::types::Boxf &bbox,
                                           std::vector<cv::Point2f> &face_landmark_5of68) {
//    InferenceTask task{input_image.clone(), bbox, face_landmark_5of68};
    auto promise = std::promise<void>();
    auto future = promise.get_future();

    // 创建任务，传入结果向量的指针
    InferenceTask task{input_image.clone(), bbox, &face_landmark_5of68, std::move(promise)};

    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        task_queue.push(std::move(task));
    }

    condition.notify_one();
}

void trt_face_68landmarks_mt::shutdown() {
    // 设置停止标志
    stop_flag = true;
    condition.notify_all();

    // 等待所有工作线程结束
    for (auto& thread : worker_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void trt_face_68landmarks_mt::wait_for_completion() {
    std::unique_lock<std::mutex> lock(completion_mutex);
    completion_cv.wait(lock, [this]() {
        return active_tasks == 0 && task_queue.empty();
    });
}

trt_face_68landmarks_mt::~trt_face_68landmarks_mt() {
    shutdown();

    // 释放CUDA资源
    for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
        for (auto buffer : buffers[thread_id]) {
            cudaFree(buffer);
        }
        cudaStreamDestroy(streams[thread_id]);
    }
}