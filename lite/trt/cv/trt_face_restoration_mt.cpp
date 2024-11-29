// trt_face_restoration_mt.cpp
#include "trt_face_restoration_mt.h"

trt_face_restoration_mt::trt_face_restoration_mt(std::string& model_path, size_t num_threads)
        : num_threads(num_threads) {
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
        worker_threads.emplace_back(&trt_face_restoration_mt::worker_function, this, i);
    }
}

// 在cpp文件中修改相关实现
void trt_face_restoration_mt::worker_function(int thread_id) {
    while (true) {
        InferenceTaskTest task;
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


void trt_face_restoration_mt::process_single_task(const InferenceTaskTest& task, int thread_id) {
    auto ori_image = task.face_swap_image.clone();

    // 1. 图像预处理
    cv::Mat crop_image;
    cv::Mat affine_matrix;
    std::tie(crop_image, affine_matrix) = face_utils::warp_face_by_face_landmark_5(
            task.face_swap_image,
            task.target_landmarks_5,
            face_utils::FFHQ_512
    );

    std::vector<float> crop_size = {512, 512};
    cv::Mat box_mask = face_utils::create_static_box_mask(crop_size);
    std::vector<cv::Mat> crop_mask_list;
    crop_mask_list.emplace_back(box_mask);

    cv::cvtColor(crop_image, crop_image, cv::COLOR_BGR2RGB);
    crop_image.convertTo(crop_image, CV_32FC3, 1.f / 255.f);
    crop_image.convertTo(crop_image, CV_32FC3, 2.0f, -1.f);

    std::vector<float> input_vector;
    trtcv::utils::transform::create_tensor(
            crop_image,
            input_vector,
            input_node_dims,
            trtcv::utils::transform::CHW
    );

    // 2. 拷贝输入数据到GPU
    cudaMemcpyAsync(
            buffers[thread_id][0],
            input_vector.data(),
            1 * 3 * 512 * 512 * sizeof(float),
            cudaMemcpyHostToDevice,
            streams[thread_id]
    );

    // 3. 同步并推理
    cudaStreamSynchronize(streams[thread_id]);
    bool status = trt_contexts[thread_id]->enqueueV3(streams[thread_id]);

    if (!status) {
        std::cerr << "Failed to inference in thread " << thread_id << std::endl;
        return;
    }

    cudaStreamSynchronize(streams[thread_id]);

    // 4. 获取输出数据
    std::vector<float> output_vector(1 * 3 * 512 * 512);
    cudaMemcpyAsync(
            output_vector.data(),
            buffers[thread_id][1],
            1 * 3 * 512 * 512 * sizeof(float),
            cudaMemcpyDeviceToHost,
            streams[thread_id]
    );

    cudaStreamSynchronize(streams[thread_id]);

    // 5. 后处理
    int channel = 3;
    int height = 512;
    int width = 512;
    std::vector<float> output(channel * height * width);
    output.assign(output_vector.begin(), output_vector.end());

    // 值范围裁剪到[-1, 1]
    std::transform(output.begin(), output.end(), output.begin(),
                   [](double x) { return std::max(-1.0, std::min(1.0, x)); });

    // 转换到[0, 1]范围
    std::transform(output.begin(), output.end(), output.begin(),
                   [](double x) { return (x + 1.f) / 2.f; });

    // CHW到HWC转换
    std::vector<float> transposed_data(channel * height * width);
    for (int c = 0; c < channel; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int src_index = c * (height * width) + h * width + w;
                int dst_index = h * (width * channel) + w * channel + c;
                transposed_data[dst_index] = output[src_index];
            }
        }
    }

    // 转换到0-255范围
    std::transform(transposed_data.begin(), transposed_data.end(), transposed_data.begin(),
                   [](float x) { return std::round(x * 255.f); });

    // 转换到uint8
    std::transform(transposed_data.begin(), transposed_data.end(), transposed_data.begin(),
                   [](float x) { return static_cast<uint8_t>(x); });

    // 6. 创建输出图像
    cv::Mat mat(height, width, CV_32FC3, transposed_data.data());
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);

    // 7. 后处理和保存
    auto crop_mask = crop_mask_list[0];
    cv::Mat paste_frame = face_utils::paste_back(ori_image, mat, crop_mask, affine_matrix);
    cv::Mat dst_image = face_utils::blend_frame(ori_image, paste_frame);
    cv::imwrite(task.face_enchaner_path, dst_image);
}

void trt_face_restoration_mt::detect_async(
        cv::Mat& face_swap_image,
        std::vector<cv::Point2f>& target_landmarks_5,
        const std::string& face_enchaner_path
) {
    InferenceTaskTest task{face_swap_image.clone(), target_landmarks_5, face_enchaner_path};

    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        task_queue.push(std::move(task));
    }

    condition.notify_one();
}


void trt_face_restoration_mt::shutdown() {
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

void trt_face_restoration_mt::wait_for_completion() {
    std::unique_lock<std::mutex> lock(completion_mutex);
    completion_cv.wait(lock, [this]() {
        return active_tasks == 0 && task_queue.empty();
    });
}

trt_face_restoration_mt::~trt_face_restoration_mt() {
    shutdown();

    // 释放CUDA资源
    for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
        for (auto buffer : buffers[thread_id]) {
            cudaFree(buffer);
        }
        cudaStreamDestroy(streams[thread_id]);
    }
}