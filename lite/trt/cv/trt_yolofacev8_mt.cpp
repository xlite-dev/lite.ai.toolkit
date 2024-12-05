//
// Created by root on 12/3/24.
//

#include "trt_yolofacev8_mt.h"
trt_yolofacev8_mt::trt_yolofacev8_mt(std::string &model_path, size_t num_threads) : num_threads(num_threads){
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
        worker_threads.emplace_back(&trt_yolofacev8_mt::worker_function, this, i);
    }

    // 6.将ratio这个修改一下
    ratio_width.resize(num_threads);
    ratio_height.resize(num_threads);
    nms_cuda_manager = std::make_unique<NMSCudaManager>();


}


std::vector<int> trt_yolofacev8_mt::nms_cuda(std::vector<lite::types::Boxf> boxes, std::vector<float> confidences,
                                             const float nms_thresh) {
    return nms_cuda_manager->perform_nms(boxes, confidences, nms_thresh);
}

void trt_yolofacev8_mt::worker_function(int thread_id) {
    while (true){
        InferenceTaskFace task;
        bool has_task = false;
        // 从任务队列中获取任务
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (!task_queue.empty()){
                // 获取队列的头任务
                task = std::move(task_queue.front());
                // 将队列的头任务弹出
                task_queue.pop();
                has_task = true;
                active_tasks++;
            } else if (stop_flag){
                // 如果出现停止标识
                break;
            } else{
                condition.wait(lock);
                continue;
            }
        }
        if (has_task){
            // 如果有任务
            process_single_task(task,thread_id);
            // 更新活跃任务计数
            {
                std::lock_guard<std::mutex> lock(completion_mutex);
                active_tasks--;
                completion_cv.notify_all();
            }
        }
    }
}

cv::Mat trt_yolofacev8_mt::normalize(cv::Mat srcimg, int thread_id) {
    const int height = srcimg.rows;
    const int width = srcimg.cols;
    cv::Mat temp_image = srcimg.clone();
    int input_height = input_node_dims[2];
    int input_width = input_node_dims[3];

    if (height > input_height || width > input_width)
    {
        const float scale = std::min((float)input_height / height, (float)input_width / width);
        cv::Size new_size = cv::Size(int(width * scale), int(height * scale));
        cv::resize(srcimg, temp_image, new_size);
    }

    ratio_height[thread_id] = (float)height / temp_image.rows;
    ratio_width[thread_id] = (float)width / temp_image.cols;

    cv::Mat input_img;
    cv::copyMakeBorder(temp_image, input_img, 0, input_height - temp_image.rows,
                       0, input_width - temp_image.cols, cv::BORDER_CONSTANT, 0);

    std::vector<cv::Mat> bgrChannels(3);
    cv::split(input_img, bgrChannels);
    for (int c = 0; c < 3; c++)
    {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / 128.0, -127.5 / 128.0);
    }
    cv::Mat normalized_image;
    cv::merge(bgrChannels,normalized_image);
    return normalized_image;
}

void trt_yolofacev8_mt::generate_box(float *trt_outputs, std::vector<lite::types::Boxf> &boxes, float conf_threshold,
                                     float iou_threshold,int thread_id) {
    int num_box = output_node_dims[0][2];

    // 直接分配目标类型的向量
    std::vector<lite::types::BoundingBoxType<float, float>> bounding_box_raw(num_box);

    // 调用包装函数
    launch_yolov8_postprocess(
            static_cast<float*>(buffers[thread_id][1]),
            num_box,
            conf_threshold,
            ratio_height[thread_id],
            ratio_width[thread_id],
            bounding_box_raw.data(),
            num_box
    );

    std::vector<float> score_raw;
    for (const auto& bbox : bounding_box_raw) {
        if (bbox.score >= 0) {
            score_raw.emplace_back(bbox.score);
        }
    }



    std::vector<int> keep_inds = nms_cuda(bounding_box_raw, score_raw, iou_threshold);
//    std::vector<int> keep_inds = this->nms(bounding_box_raw, score_raw, iou_threshold);

    const int keep_num = keep_inds.size();
    boxes.clear();
    boxes.resize(keep_num);
    for (int i = 0; i < keep_num; i++)
    {
        const int ind = keep_inds[i];
        boxes[i] = bounding_box_raw[ind];
    }

}

float trt_yolofacev8_mt::get_iou(const lite::types::Boxf box1, const lite::types::Boxf box2) {

}

void trt_yolofacev8_mt::process_single_task( InferenceTaskFace &task, int thread_id) {
    if (task.input_mat.empty()) return;
    // 检查 TRT 上下文
    if (!trt_contexts[thread_id]) {
        std::cerr << "TensorRT context is null!" << std::endl;
        return;
    }

    // 计算preprocess的时间
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat normalized_image = normalize(task.input_mat,thread_id);

    // 2.trans to input vector
    std::vector<float> input;
    trtcv::utils::transform::create_tensor(normalized_image,input,input_node_dims,trtcv::utils::transform::CHW);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
    std::cout << "Face Detect preprocess time: " << duration.count() / 1000 << "ms" << std::endl;

    // 3. infer
    // 计算推理时间
    auto start_infer = std::chrono::high_resolution_clock::now();
    cudaMemcpyAsync(buffers[thread_id][0], input.data(), input_node_dims[0] * input_node_dims[1] * input_node_dims[2] * input_node_dims[3] * sizeof(float),
                    cudaMemcpyHostToDevice, streams[thread_id]);
    bool status = trt_contexts[thread_id]->enqueueV3(streams[thread_id]);
    cudaStreamSynchronize(streams[thread_id]);
    auto end_infer = std::chrono::high_resolution_clock::now();
    auto duration_infer = std::chrono::duration_cast<std::chrono::microseconds>(end_infer-start_infer);
    std::cout << "Face Detect infer time: " << duration_infer.count() / 1000 << "ms" << std::endl;


    if (!status){
        std::cerr << "Failed to infer by TensorRT." << std::endl;
        return;
    }

    std::vector<float> output(output_node_dims[0][0] * output_node_dims[0][1] * output_node_dims[0][2]);

    cudaMemcpyAsync(output.data(), buffers[thread_id][1], output_node_dims[0][0] * output_node_dims[0][1] * output_node_dims[0][2] * sizeof(float),
                    cudaMemcpyDeviceToHost, streams[thread_id]);
    // 4. generate box
    // 计算后处理时间
    auto start_post = std::chrono::high_resolution_clock::now();
    generate_box(output.data(),*task.bbox,0.45f,0.5f,thread_id);
    auto end_post = std::chrono::high_resolution_clock::now();
    auto duration_post = std::chrono::duration_cast<std::chrono::microseconds>(end_post-start_post);
    std::cout << "Face Detect postprocess time: " << duration_post.count() / 1000 << "ms" << std::endl;

}

void trt_yolofacev8_mt::detect_async(cv::Mat &input_image, std::vector<lite::types::Boxf>* bbox
                                    ) {
    //    InferenceTask task{input_image.clone(), bbox, face_landmark_5of68};
    auto promise = std::promise<void>();
    auto future = promise.get_future();

    // 创建任务，传入结果向量的指针
    InferenceTaskFace task{input_image.clone(), bbox,  std::move(promise)};

    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        task_queue.push(std::move(task));
    }

    condition.notify_one();
}

void trt_yolofacev8_mt::shutdown() {
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

void trt_yolofacev8_mt::wait_for_completion() {
    std::unique_lock<std::mutex> lock(completion_mutex);
    completion_cv.wait(lock, [this]() {
        return active_tasks == 0 && task_queue.empty();
    });
}

trt_yolofacev8_mt::~trt_yolofacev8_mt() {
    // 释放CUDA流
    for (size_t i = 0; i < num_threads; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    // 释放CUDA缓冲区
    for (size_t i = 0; i < num_threads; ++i) {
        for (auto& buffer : buffers[i]) {
            cudaFree(buffer);
        }
    }
}