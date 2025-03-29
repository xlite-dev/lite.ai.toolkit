//
// Created by wangzijian on 3/5/25.
//

#include "trt_facefusion_pipeline_gpu.h"



trt_facefusion_pipeline_gpu::trt_facefusion_pipeline_gpu(const std::vector<std::string> &model_path_list) {
    // 加载四个模型
    // 初始化全局 runtime
    trt_runtime = std::unique_ptr<IRuntime>(createInferRuntime(trt_logger));
    engines.resize(model_path_list.size());
    contexts.resize(model_path_list.size());
    model_ios.resize(model_path_list.size());
    input_dims_list.resize(model_path_list.size());
    output_dims_list.resize(model_path_list.size());

    input_dims_list[0].resize(4);
    input_dims_list[0][0] = 1;
    input_dims_list[0][1] = 3;
    input_dims_list[0][2] = 640;
    input_dims_list[0][3] = 640;

    for (int i = 0; i < model_path_list.size(); ++i) {
        init_engine(model_path_list[i], i);
    }
    // 创建全局 Stream
    cudaStreamCreate(&stream);

    // 预分配中间结果内存（示例：假设最大支持10张人脸）
    cudaMalloc(&d_bboxes, 10 * 4 * sizeof(float));      // 每张人脸4个坐标
    cudaMalloc(&d_landmarks, 10 * 68 * 2 * sizeof(float)); // 每张人脸68个关键点
    cudaMalloc(&d_swapped, 640 * 640 * 3 * sizeof(float)); // 换脸结果
    cudaMalloc(&d_final, 640 * 640 * 3 * sizeof(float));   // 最终输出

}

void trt_facefusion_pipeline_gpu::init_engine(const std::string &model_path, int model_idx) {
    std::ifstream file(model_path.data(), std::ios::binary);

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

    trt_runtime.reset(nvinfer1::createInferRuntime(trt_logger));
    // 对vector的每个都进行初始化和赋值
    engines[model_idx].reset(trt_runtime->deserializeCudaEngine(model_data.data(), model_size));
    contexts[model_idx].reset(engines[model_idx]->createExecutionContext());

    auto num_io_tensors = engines[model_idx]->getNbIOTensors();
    // 分别计算输入和输出张量的数量
    int nbInputs = 0;
    int nbOutputs = 0;

    for (int i = 0; i < num_io_tensors; i++) {
        const char* name = engines[model_idx]->getIOTensorName(i);
        if (engines[model_idx]->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            nbInputs++;
        } else {
            nbOutputs++;
        }
    }


    // 判断是num_io是否大于2
    if (nbInputs > 1)
    {
        // 输入是name的前两个元素
        for (int i=0 ; i < num_io_tensors; i++)
        {
            auto current_input_name = engines[model_idx]->getIOTensorName(i);
            auto current_input_dims =  engines[model_idx]->getTensorShape(current_input_name);
            model_ios[model_idx].input_size = model_ios[model_idx].input_size + volume(current_input_dims);
            if (i == 2)
            {
                auto current_output_name = engines[model_idx]->getIOTensorName(i);
                auto current_output_dims =  engines[model_idx]->getTensorShape(current_output_name);
                model_ios[model_idx].output_size =  volume(current_output_dims);
            }
        }

    }else{

        if (nbOutputs < 2)
        {
            // 这里处理大多数的输入的时候
            auto input_name = engines[model_idx]->getIOTensorName(0);
            auto output_name = engines[model_idx]->getIOTensorName(1);
            auto input_dims = engines[model_idx]->getTensorShape(input_name);
            auto output_dims = engines[model_idx]->getTensorShape(output_name);

            model_ios[model_idx].input_size = volume(input_dims);
            model_ios[model_idx].output_size = volume(output_dims);
        }else{
            // 处理两个输出的算法
            // 输入是name的前两个元素
            for (int i=0 ; i < num_io_tensors; i++)
            {
                if (i == 0)
                {
                    auto current_input_name = engines[model_idx]->getIOTensorName(i);
                    auto current_output_dims =  engines[model_idx]->getTensorShape(current_input_name);
                    model_ios[model_idx].input_size =  volume(current_output_dims);
                }else{
                    auto current_output_name = engines[model_idx]->getIOTensorName(i);
                    auto current_output_dims =  engines[model_idx]->getTensorShape(current_output_name);
                    model_ios[model_idx].output_size = model_ios[model_idx].output_size + volume(current_output_dims);
                }
            }
        }
    }

    std::cout<<"hello"<<std::endl;
    // 分配gpu上的空间大小

    CHECK_CUDA(cudaMalloc(&model_ios[model_idx].d_input, model_ios[model_idx].input_size * sizeof(float )););
    CHECK_CUDA(cudaMalloc(&model_ios[model_idx].d_output, model_ios[model_idx].output_size * sizeof(float )););
}

trt_facefusion_pipeline_gpu::~trt_facefusion_pipeline_gpu() {
    // 释放引擎和上下文
    for (auto& ctx : contexts) ctx.reset();
    for (auto& engine : engines) engine.reset();
    trt_runtime.reset();

    // 释放 GPU 内存
    for (auto& io : model_ios) {
        cudaFree(io.d_input);
        cudaFree(io.d_output);
    }
    cudaFree(d_bboxes);
    cudaFree(d_landmarks);
    cudaFree(d_swapped);
    cudaFree(d_final);

    // 销毁 Stream
    cudaStreamDestroy(stream);
}


// CUDA kernel for resizing
__global__ void resizeKernel(const unsigned char* input, unsigned char* output,
                             int src_width, int src_height, int dst_width, int dst_height,
                             int channels, int src_stride, int dst_stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dst_width && y < dst_height) {
        float src_x = x * (float)src_width / dst_width;
        float src_y = y * (float)src_height / dst_height;

        // Simple nearest neighbor for simplicity (can be improved to bilinear)
        int src_x_int = (int)src_x;
        int src_y_int = (int)src_y;

        for (int c = 0; c < channels; c++) {
            output[y * dst_stride + x * channels + c] =
                    input[src_y_int * src_stride + src_x_int * channels + c];
        }
    }
}

// CUDA kernel for BGR normalization (and padding)
__global__ void normalizeKernel(const unsigned char* input, float* output,
                                int input_width, int input_height,
                                int output_width, int output_height,
                                int channels, int input_stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < output_width && y < output_height) {
        for (int c = 0; c < channels; c++) {
            // If within the actual image area, normalize; otherwise use padding (0)
            float val = 0.0f;
            if (x < input_width && y < input_height) {
                val = input[y * input_stride + x * channels + c] / 128.0f - 127.5f / 128.0f;
            } else {
                val = -127.5f / 128.0f; // Padding value normalized
            }

            // Store in NCHW format typically expected by neural networks
            output[c * output_height * output_width + y * output_width + x] = val;
        }
    }
}



// Version that outputs directly to a preallocated GPU buffer
// Useful for direct integration with TensorRT
void trt_facefusion_pipeline_gpu::normalize_cuda_direct(cv::Mat srcimg, float* output_gpu_buffer) {
    const int height = srcimg.rows;
    const int width = srcimg.cols;
    int input_height = input_dims_list[0][2];
    int input_width = input_dims_list[0][3];

    // Calculate resize dimensions
    int resize_height, resize_width;
    if (height > input_height || width > input_width) {
        const float scale = std::min((float)input_height / height, (float)input_width / width);
        resize_width = int(width * scale);
        resize_height = int(height * scale);
    } else {
        resize_width = width;
        resize_height = height;
    }

    // Update resize ratios
    ratio_height = (float)height / resize_height;
    ratio_width = (float)width / resize_width;

    // Allocate CUDA memory for intermediate steps
    unsigned char* d_srcimg;
    unsigned char* d_resized;

    cudaMalloc(&d_srcimg, width * height * 3 * sizeof(unsigned char));
    cudaMalloc(&d_resized, resize_width * resize_height * 3 * sizeof(unsigned char));

    // Copy input image to GPU
    cudaMemcpy(d_srcimg, srcimg.data, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Resize kernel
    dim3 resizeBlock(16, 16);
    dim3 resizeGrid((resize_width + resizeBlock.x - 1) / resizeBlock.x,
                    (resize_height + resizeBlock.y - 1) / resizeBlock.y);

    resizeKernel<<<resizeGrid, resizeBlock>>>(
            d_srcimg, d_resized,
            width, height, resize_width, resize_height,
            3, width * 3, resize_width * 3
    );

    // Normalize and pad kernel
    dim3 normalizeBlock(16, 16);
    dim3 normalizeGrid((input_width + normalizeBlock.x - 1) / normalizeBlock.x,
                       (input_height + normalizeBlock.y - 1) / normalizeBlock.y);

    normalizeKernel<<<normalizeGrid, normalizeBlock>>>(
            d_resized, output_gpu_buffer,
            resize_width, resize_height, input_width, input_height,
            3, resize_width * 3
    );

    // Free CUDA memory for intermediate steps
    cudaFree(d_srcimg);
    cudaFree(d_resized);

    // No need to copy back to CPU as we're writing directly to the provided GPU buffer
}

// ===============================================================================================
// 上面是预处理的Kern
// =============================================================================================




// 修改后的yolov8_gpu_postprocessor.cu，支持返回GPU上的结果指针

#include <algorithm>

// CUDA核心用于初始bbox解码
__global__ void yolov8_decode_kernel(
        float* detection_output,
        int num_boxes,
        float conf_threshold,
        float ratio_height,
        float ratio_width,
        lite::types::Boxf* boxes,
        int* valid_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;

    const float score = detection_output[4 * num_boxes + idx];

    if (score > conf_threshold) {
        // 计算坐标
        float x_center = detection_output[idx];
        float y_center = detection_output[num_boxes + idx];
        float width = detection_output[2 * num_boxes + idx];
        float height = detection_output[3 * num_boxes + idx];

        // 转换为角点格式并应用缩放
        float x1 = (x_center - 0.5f * width) * ratio_width;
        float y1 = (y_center - 0.5f * height) * ratio_height;
        float x2 = (x_center + 0.5f * width) * ratio_width;
        float y2 = (y_center + 0.5f * height) * ratio_height;

        // 使用原子操作获取输出索引
        int output_idx = atomicAdd(valid_count, 1);

        // 存储bbox
        boxes[output_idx].x1 = x1;
        boxes[output_idx].y1 = y1;
        boxes[output_idx].x2 = x2;
        boxes[output_idx].y2 = y2;
        boxes[output_idx].score = score;
        boxes[output_idx].flag = true;
    }
}

// 计算两个框之间的IoU (GPU设备函数)
__device__ float calculate_iou(const lite::types::Boxf& box1, const lite::types::Boxf& box2) {
    float xx1 = max(box1.x1, box2.x1);
    float yy1 = max(box1.y1, box2.y1);
    float xx2 = min(box1.x2, box2.x2);
    float yy2 = min(box1.y2, box2.y2);

    float w = max(0.0f, xx2 - xx1);
    float h = max(0.0f, yy2 - yy1);
    float inter_area = w * h;

    float box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    float box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);

    float union_area = box1_area + box2_area - inter_area;

    return union_area > 0.0f ? inter_area / union_area : 0.0f;
}

// 仅在GPU上进行简单的冒泡排序
__global__ void bubble_sort_kernel(lite::types::Boxf* boxes, int count) {
    // 对于小数据集的简单排序，单线程就足够了
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < count - i - 1; j++) {
                if (boxes[j].score < boxes[j + 1].score) {
                    // 交换
                    lite::types::Boxf temp = boxes[j];
                    boxes[j] = boxes[j + 1];
                    boxes[j + 1] = temp;
                }
            }
        }
    }
}

// 在GPU上执行NMS
__global__ void nms_kernel(
        lite::types::Boxf* boxes,
        int num_boxes,
        float nms_threshold,
        bool* keep
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;

    // 如果这个框已经被标记为删除，直接返回
    if (!keep[idx]) return;

    for (int i = 0; i < num_boxes; i++) {
        // 跳过自身和已经被标记为删除的框
        if (i == idx || !keep[i]) continue;

        // 仅检查得分更高的框
        if (i < idx) {
            // 如果当前框与得分更高的框有高IoU，标记为删除
            if (calculate_iou(boxes[idx], boxes[i]) > nms_threshold) {
                keep[idx] = false;
                break;
            }
        }
    }
}

// 将保留的框收集到连续内存中
__global__ void collect_results_kernel(
        lite::types::Boxf* input_boxes,
        bool* keep,
        int num_boxes,
        lite::types::Boxf* output_boxes,
        int* output_count
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;

    if (keep[idx]) {
        int out_idx = atomicAdd(output_count, 1);
        output_boxes[out_idx] = input_boxes[idx];
    }
}

// 释放GPU上分配的输出框内存
void free_gpu_output_boxes(lite::types::Boxf* d_boxes) {
    if (d_boxes != nullptr) {
        cudaFree(d_boxes);
    }
}

// 更新的主包装函数实现
void trt_facefusion_pipeline_gpu::process_yolov8_detections_gpu(
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
) {
    if (stream == nullptr) {
        cudaStreamCreate(&stream);
    }

    // 为输出框和有效计数分配设备内存
    lite::types::Boxf* d_boxes;
    int* d_valid_count;
    cudaMalloc(&d_boxes, num_boxes * sizeof(lite::types::Boxf));
    cudaMalloc(&d_valid_count, sizeof(int));
    cudaMemset(d_valid_count, 0, sizeof(int));

    // 计算解码内核的启动参数
    int block_size = 256;
    int grid_size = (num_boxes + block_size - 1) / block_size;

    // 启动解码内核
    yolov8_decode_kernel<<<grid_size, block_size, 0, stream>>>(
            face_detect_infer,
            num_boxes,
            conf_threshold,
            ratio_height,
            ratio_width,
            d_boxes,
            d_valid_count
    );

    // 将有效计数复制回主机
    int valid_count;
    cudaMemcpyAsync(&valid_count, d_valid_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 如果没有有效框，提前退出
    if (valid_count == 0) {
        output_boxes.clear();
        if (d_output_boxes_ptr != nullptr) {
            *d_output_boxes_ptr = nullptr;
        }
        if (output_box_count != nullptr) {
            *output_box_count = 0;
        }
        cudaFree(d_boxes);
        cudaFree(d_valid_count);
        return;
    }

    // 在GPU上对框进行排序（从高分到低分）
    bubble_sort_kernel<<<1, 1, 0, stream>>>(d_boxes, valid_count);

    // 分配用于NMS的设备内存
    bool* d_keep;
    cudaMalloc(&d_keep, valid_count * sizeof(bool));
    cudaMemset(d_keep, 1, valid_count * sizeof(bool)); // 默认保留所有框

    // 执行NMS
    nms_kernel<<<grid_size, block_size, 0, stream>>>(
            d_boxes,
            valid_count,
            iou_threshold,
            d_keep
    );

    // 收集结果
    lite::types::Boxf* d_output_boxes;
    int* d_output_count;
    cudaMalloc(&d_output_boxes, valid_count * sizeof(lite::types::Boxf));
    cudaMalloc(&d_output_count, sizeof(int));
    cudaMemset(d_output_count, 0, sizeof(int));

    collect_results_kernel<<<grid_size, block_size, 0, stream>>>(
            d_boxes,
            d_keep,
            valid_count,
            d_output_boxes,
            d_output_count
    );

    // 获取输出框的数量
    int output_count;
    cudaMemcpyAsync(&output_count, d_output_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 将结果拷贝回CPU（如果需要）
    if (!output_boxes.empty() || output_boxes.size() != output_count) {
        output_boxes.resize(output_count);
    }

    if (!output_boxes.empty()) {
        cudaMemcpyAsync(output_boxes.data(), d_output_boxes, output_count * sizeof(lite::types::Boxf),
                        cudaMemcpyDeviceToHost, stream);
    }

    // 如果请求了GPU指针，则将其返回给调用方，并不释放内存
    if (d_output_boxes_ptr != nullptr) {
        *d_output_boxes_ptr = d_output_boxes;
    } else {
        // 否则释放内存
        cudaFree(d_output_boxes);
    }

    // 返回框的数量
    if (output_box_count != nullptr) {
        *output_box_count = output_count;
    }

    // 清理其他资源
    cudaFree(d_boxes);
    cudaFree(d_valid_count);
    cudaFree(d_keep);
    cudaFree(d_output_count);

    cudaStreamSynchronize(stream);
}


void trt_facefusion_pipeline_gpu::detect(cv::Mat &src_image, cv::Mat &target_image, int idx_of_src, int idx_of_target) {
    // 先开辟好一个cv::Mat所需要的空间
    float* output_gpu_buffer;
    float* h_output;
    int input_height =640;
    int input_width = 640;
    int buffer_size = input_height * input_width * 3 * sizeof(float);  // 3 表示 RGB 通道数
    //在cpu上分配内存
    h_output = (float *) malloc(buffer_size);
    // 在 GPU 上分配内存
    cudaMalloc(&output_gpu_buffer, buffer_size);

    // 调用 normalize_cuda_direct 函数
    normalize_cuda_direct(src_image, output_gpu_buffer);

    // 测试前处理的结果
//    cudaMemcpy(h_output,output_gpu_buffer,buffer_size,cudaMemcpyDeviceToHost);
//
//    std::vector<float> normalized{h_output,h_output+ (3 * input_height  * input_width)};

    // 第一个模型(yolov8face)推理 - 检测人脸
    // 将处理后的图像数据复制到模型的输入内存
    // 这里是全局的
    cudaMemcpyAsync(output_gpu_buffer, output_gpu_buffer, 640 *640 *3 *sizeof(float ),
                    cudaMemcpyDeviceToDevice, stream);

    float* face_detect_infer;
    cudaMalloc(&face_detect_infer,20 * 8400 * sizeof(float ));

    contexts[1]->setTensorAddress("output0",face_detect_infer);
    contexts[1]->setTensorAddress("images", output_gpu_buffer);
    bool status = contexts[1]->enqueueV3(stream);
    if (!status){
        std::cerr << "Failed to infer by TensorRT." << std::endl;
        return;
    }

    // 测试gpu上的数据是否正确
//    std::vector<float> output(1 * 20 * 8400);
//    cudaMemcpyAsync(output.data(),face_detect_infer,20 * 8400 *sizeof (float ),cudaMemcpyDeviceToHost);


    int num_of_boxes = 8400;
    std::vector<lite::types::Boxf> boxes;
    std::vector<lite::types::Boxf> boxes_gpu_test;




    // 声明GPU输出指针和框数量
    lite::types::Boxf* d_output_boxes = nullptr;
    int output_box_count = 0;


    process_yolov8_detections_gpu(face_detect_infer,num_of_boxes,0.45f,0.5f,ratio_height,ratio_width,
                                  boxes,stream,&d_output_boxes,  // 接收GPU上的结果指针
                                  &output_box_count);


    // Resize the vector to match the output count BEFORE the memory copy
    boxes_gpu_test.resize(output_box_count);

// Now that the vector is properly sized, copy the data
    if (output_box_count > 0) {
        cudaMemcpyAsync(boxes_gpu_test.data(), d_output_boxes, output_box_count * sizeof(lite::types::Boxf),
                        cudaMemcpyDeviceToHost, stream);

        // Make sure the copy is complete before proceeding
        cudaStreamSynchronize(stream);
    }

    cv::Mat temp = cv::imread("/home/lite.ai.toolkit/1.jpg");

    lite::utils::draw_boxes_inplace(temp, boxes_gpu_test);
    cv::imwrite("/home/lite.ai.toolkit/1_gpu_version.jpg",temp);


    detect_landmarks_68(d_output_boxes,);


    std::cout<<"trt face detect done! has "<< boxes.size() << " face"<<std::endl;

    std::cout<<"hello";
}


void trt_facefusion_pipeline_gpu::detect_landmarks_68(lite::types::Boxf *d_output_boxes, int *d_number_of_boxes) {

}