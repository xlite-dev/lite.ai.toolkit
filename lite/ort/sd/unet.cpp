//
// Created by wangzijian on 8/14/24.
//

#include "unet.h"
#include <random>
#include <cstdint>
#include <iomanip>
#include "tokenizer.h"
#include "half.h"
using ortsd::UNet;

uint16_t float32_to_float16(float value) {
    uint32_t x = *reinterpret_cast<uint32_t*>(&value);
    uint16_t sign = (x >> 16) & 0x8000;
    uint16_t exponent = ((x >> 23) & 0xff) - (127 - 15);
    uint16_t mantissa = (x >> 13) & 0x3ff;

    if (exponent <= 0) {
        if (exponent < -10) {
            return sign; // 太小，返回0
        }
        mantissa = (mantissa | 0x400) >> (1 - exponent);
        return sign | mantissa;
    } else if (exponent == 0xff - (127 - 15)) {
        if (mantissa == 0) {
            return sign | 0x7c00; // 无穷大
        } else {
            return sign | 0x7c00 | (mantissa >> 13); // NaN
        }
    } else if (exponent > 30) {
        return sign | 0x7c00; // 溢出为无穷大
    }

    return sign | (exponent << 10) | mantissa;
}

// 辅助函数：将FP16转换回FP32
float float16_to_float32(uint16_t h) {
    unsigned int sign = ((h >> 15) & 1);
    unsigned int exponent = ((h >> 10) & 0x1f);
    unsigned int mantissa = ((h & 0x3ff) << 13);

    if (exponent == 0x1f) {  // Inf or NaN
        if (mantissa == 0) {
            return sign ? -INFINITY : INFINITY;
        } else {
            return NAN;
        }
    }

    if (exponent == 0) {  // Subnormal or zero
        if (mantissa == 0) {
            return sign ? -0.0f : 0.0f;
        } else {
            float f = (float)mantissa / (1 << 23);
            return sign ? -f : f;
        }
    }

    unsigned int f = ((sign << 31) | ((exponent + 112) << 23) | mantissa);
    return *reinterpret_cast<float*>(&f);
}

void PrintModelInputInfo(Ort::Session* session, const Ort::AllocatorWithDefaultOptions& allocator) {
    size_t num_input_nodes = session->GetInputCount();

    std::cout << "Number of inputs = " << num_input_nodes << std::endl;

    for (size_t i = 0; i < num_input_nodes; ++i) {
        auto input_name = session->GetInputNameAllocated(i, allocator);
        Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        std::vector<int64_t> input_node_dims = tensor_info.GetShape();

        std::cout << "Input " << i << " : name = " << input_name.get() << std::endl;
        std::cout << "Input " << i << " : type = " << type << std::endl;
        std::cout << "Input " << i << " : num_dims = " << input_node_dims.size() << std::endl;

        for (size_t j = 0; j < input_node_dims.size(); ++j) {
            std::cout << "Input " << i << " : dim[" << j << "] = " << input_node_dims[j] << std::endl;
        }
    }
}


UNet::UNet(const std::string &_onnx_path, unsigned int _num_threads)  :
        log_id(_onnx_path.data()), num_threads(_num_threads){
    std::cout<<"unet test"<<std::endl;
    onnx_path = _onnx_path.data();
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(num_threads);
    session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_options.SetLogSeverityLevel(4);
    ort_session = new Ort::Session(ort_env, onnx_path, session_options);
    std::cout << "Load " << onnx_path << " done!" << std::endl;


    PrintModelInputInfo(ort_session, allocator);


}


UNet::~UNet() {
    if (ort_session)
        delete ort_session;
    ort_session = nullptr;
}

void generate_latents(std::vector<float>& latents, int batch_size, int unet_channels, int latent_height, int latent_width, float init_noise_sigma) {
    // 计算总大小
    size_t total_size = batch_size * unet_channels * latent_height * latent_width;

    // 初始化随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f); // 正态分布，均值为0，标准差为1

    // 生成随机数
    for (size_t i = 0; i < total_size; ++i) {
        latents[i] = dist(gen) * init_noise_sigma;
    }
}

void save_tensor_to_file(const Ort::Value& tensor, const std::string& filename) {
    // 获取张量的类型和形状信息
    auto type_and_shape_info = tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = type_and_shape_info.GetShape();
    size_t element_count = type_and_shape_info.GetElementCount();

    ONNXTensorElementDataType type = type_and_shape_info.GetElementType();
    if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        std::cerr << "Unsupported tensor data type. Only float tensors are supported." << std::endl;
        return;
    }

    const uint16_t* pdata = tensor.GetTensorData<uint16_t>();

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open file for writing: " << filename << std::endl;
        return;
    }

    file << std::setprecision(9);  // 设置精度以确保准确性
    for (size_t i = 0; i < element_count; ++i) {
        float f32_value = float16_to_float32(pdata[i]);
        file << f32_value << "\n";
    }
    file.close();
}



void UNet::inference(std::vector<std::string> input, std::vector<std::vector<float>> &output) {
    // generate noise latent image init
    auto scheduler = Scheduler::DDIMScheduler("/home/lite.ai.toolkit/lite/ort/sd/scheduler_config.json");

    scheduler.set_timesteps(30);

    std::vector<int> timesteps;
    scheduler.get_timesteps(timesteps);

    auto init_noise_sigma = scheduler.get_init_noise_sigma();

    std::vector<float> latents(1 * 4 * 64 * 64);

    generate_latents(latents,1,4,64,64,init_noise_sigma);

    // copy one latent to end
    // 初始化latents中的元素为某个值（例如，1.0），仅为示例
    latents.insert(latents.end(),latents.begin(),latents.end());

    // 转换 latents 到 Float16_t
    std::vector<uint16_t> latents_fp16(latents.size());
    for (size_t i = 0; i < latents.size(); ++i) {
        latents_fp16[i] = float32_to_float16(latents[i]);
    }

    Ort::MemoryInfo allocator_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // make sample input
    std::vector<int64_t> input_node_dims = {2, 4, 64, 64};
//    inline Value Value::CreateTensor(const OrtMemoryInfo* info, void* p_data, size_t p_data_byte_count, const int64_t* shape, size_t shape_len,
//                                     ONNXTensorElementDataType type

    // 计算 latents 数据的字节大小
    size_t latents_size_in_bytes = latents_fp16.size() * sizeof(uint16_t);
    // 使用重载的 CreateTensor 函数来创建 Tensor
    Ort::Value inputTensor_latent = Ort::Value::CreateTensor(
            allocator_info,
            latents_fp16.data(),                   // 数据指针
            latents_size_in_bytes,            // 数据大小（字节）
            input_node_dims.data(),           // 形状指针
            input_node_dims.size(),           // 形状长度
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16  // 数据类型
    );

    save_tensor_to_file(inputTensor_latent,"/home/lite.ai.toolkit/input_latent.txt");

    std::vector<int64_t> input_node_dims1 = {1};
    // 转换 time_step 到 FP16
    std::vector<float> time_step = {952.0f};
    std::vector<uint16_t> time_step_fp16(time_step.size());
    for (size_t i = 0; i < time_step_fp16.size(); ++i) {
        time_step_fp16[i] = float32_to_float16(time_step[i]);
    }


    size_t time_step_size_in_bytes = time_step_fp16.size() * sizeof(uint16_t);

    Ort::Value inputTensor_timestep = Ort::Value::CreateTensor(
            allocator_info,
            time_step_fp16.data(),                   // 数据指针
            time_step_size_in_bytes,            // 数据大小（字节）
            input_node_dims1.data(),           // 形状指针
            input_node_dims1.size(),           // 形状长度
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16  // 数据类型
    );

    save_tensor_to_file(inputTensor_timestep,"/home/lite.ai.toolkit/input_timestep.txt");


    std::string onnx_path = "/home/lite.ai.toolkit/examples/hub/onnx/sd/clip_model.onnx";
    Clip clip(onnx_path);

    std::vector<std::string> input_text = {"a beauty girl"};
    std::vector<std::string> negative_input_text = {"anime"};
    std::vector<std::vector<float>> output_embedding;
    std::vector<std::vector<float>> negative_output_embedding;

    clip.inference(input_text,output_embedding);
    clip.inference(negative_input_text,negative_output_embedding);

    size_t total_size = output_embedding.size() * output_embedding[0].size() +
                        negative_output_embedding.size() * negative_output_embedding[0].size();
    std::vector<float> combined_embedding(total_size);
    memcpy(combined_embedding.data(),output_embedding.data()->data(),
           output_embedding.size() * output_embedding[0].size() * sizeof(float));

    memcpy(combined_embedding.data() + output_embedding.size() *  output_embedding[0].size(),
           negative_output_embedding.data()->data(),
           negative_output_embedding.size() * negative_output_embedding[0].size() * sizeof(float));





    std::vector<int64_t> input_node_dims2 = {2,77,768};

    std::vector<uint16_t> combined_embedding_fp16(combined_embedding.size());
    for (size_t i = 0; i < combined_embedding_fp16.size(); ++i) {
        combined_embedding_fp16[i] = float32_to_float16(combined_embedding[i]);
    }

    size_t embedding_size = combined_embedding.size() * sizeof(uint16_t);
    Ort::Value inputTensor_embedding = Ort::Value::CreateTensor(
            allocator_info,
            combined_embedding_fp16.data(),                   // 数据指针
            embedding_size,            // 数据大小（字节）
            input_node_dims2.data(),           // 形状指针
            input_node_dims2.size(),           // 形状长度
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16  // 数据类型
    );

    save_tensor_to_file(inputTensor_embedding,"/home/lite.ai.toolkit/input_embedding.txt");

    std::vector<Ort::Value> inputTensors;
    // 使用std::move 来移动Ort的对象 因为这些对象只能被移动不能被复制
    inputTensors.emplace_back(std::move(inputTensor_latent));
    inputTensors.emplace_back(std::move(inputTensor_timestep));
    inputTensors.emplace_back(std::move(inputTensor_embedding));


    Ort::RunOptions runOptions;

    // run inference
    std::vector<Ort::Value> outputTensors = ort_session->Run(
            runOptions,
            input_node_names.data(),
            inputTensors.data(),
            inputTensors.size(),
            output_node_names.data(),
            output_node_names.size()
    );


    const uint16_t *text_feature_ptr = outputTensors[0].GetTensorMutableData<uint16_t>();
    auto shape_info = outputTensors[0].GetTensorTypeAndShapeInfo();


    size_t dims_count = shape_info.GetDimensionsCount();
    std::vector<int64_t> dims = shape_info.GetShape();

    for (size_t i = 0; i < dims_count; i++) {
        int64_t dim_size = shape_info.GetShape()[i];
        std::cout << "Dimension " << i << " size: " << dim_size << std::endl;
    }
    int64_t total_elements = shape_info.GetElementCount();

    // 确保维度正确
    if (dims.size() != 4 || dims[0] < 1) {
        std::cerr << "Unexpected tensor shape" << std::endl;
        return;
    }

    int batch = dims[0];
    int channels = dims[1];
    int height = dims[2];
    int width = dims[3];

    auto num_elements_to_copy = channels * height * width;
    auto len = batch * channels * width * height;
    std::vector<float> output_test;
    output_test.reserve(len);
    for (int i = 0; i < len; ++i) {
        output_test.push_back(float16_to_float32(text_feature_ptr[i]));
    }

    std::vector<Ort::Float16_t> noise_pred_uncond(num_elements_to_copy);
    std::vector<Ort::Float16_t> noise_pred_text(num_elements_to_copy);

    auto noise_pred_uncond_begin_ptr = text_feature_ptr;
    auto noise_pred_uncond_end_ptr = text_feature_ptr + num_elements_to_copy;

    // 将 [begin_ptr, end_ptr) 区间的数据拷贝到 noise_pred_uncond 中
    memcpy(noise_pred_uncond.data(), noise_pred_uncond_begin_ptr, (noise_pred_uncond_end_ptr - noise_pred_uncond_end_ptr) * sizeof(Ort::Float16_t));

    auto noise_pred_text_begin_ptr = text_feature_ptr + num_elements_to_copy;
    auto noise_pred_text_end_ptr = text_feature_ptr + 2 *num_elements_to_copy;
    memcpy(noise_pred_text.data(), noise_pred_text_begin_ptr, (noise_pred_text_end_ptr - noise_pred_text_begin_ptr) * sizeof(Ort::Float16_t));

}