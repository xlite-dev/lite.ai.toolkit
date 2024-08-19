//
// Created by wangzijian on 8/14/24.
//

#include "unet.h"
#include <random>
using ortsd::UNet;
#include "tokenizer.h"
#include "half.h"
using half_float::half;

// 自定义转换函数
// 自定义16位浮点数转32位浮点数的解码函数
float float16ToFloat(Ort::Float16_t h) {
    uint16_t x = *reinterpret_cast<uint16_t*>(&h);
    int sign = (x >> 15) & 1;
    int exp = (x >> 10) & 0x1f;
    int mantissa = x & 0x3ff;

    float f;
    if (exp == 0) {
        if (mantissa == 0) {
            f = sign ? -0.0f : 0.0f;
        } else {
            // 非规格化数
            exp = -14;
            while (!(mantissa & 0x200)) {
                mantissa <<= 1;
                exp--;
            }
            mantissa &= 0x1ff;
            f = (sign ? -1 : 1) * ldexpf(mantissa, exp - 10);
        }
    } else if (exp == 31) {
        f = mantissa ? std::numeric_limits<float>::quiet_NaN()
                     : (sign ? -std::numeric_limits<float>::infinity()
                             : std::numeric_limits<float>::infinity());
    } else {
        // 规格化数
        f = (sign ? -1 : 1) * ldexpf(1024 + mantissa, exp - 25);
    }

    return f;
}


Ort::Float16_t floatToFloat16(float f) {
    uint16_t h;
    uint32_t x = *reinterpret_cast<uint32_t*>(&f);
    h = ((x >> 16) & 0x8000) | ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((x >> 13) & 0x03ff);
    return *reinterpret_cast<Ort::Float16_t*>(&h);
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
    std::vector<Ort::Float16_t> latents_fp16(latents.size());
    for (size_t i = 0; i < latents.size(); ++i) {
        latents_fp16[i] = floatToFloat16(latents[i]);
    }

    Ort::MemoryInfo allocator_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // make sample input
    std::vector<int64_t> input_node_dims = {2, 4, 64, 64};
//    inline Value Value::CreateTensor(const OrtMemoryInfo* info, void* p_data, size_t p_data_byte_count, const int64_t* shape, size_t shape_len,
//                                     ONNXTensorElementDataType type

    // 计算 latents 数据的字节大小
    size_t latents_size_in_bytes = latents.size() * sizeof(float);
    // 使用重载的 CreateTensor 函数来创建 Tensor
    Ort::Value inputTensor_latent = Ort::Value::CreateTensor(
            allocator_info,
            latents.data(),                   // 数据指针
            latents_size_in_bytes,            // 数据大小（字节）
            input_node_dims.data(),           // 形状指针
            input_node_dims.size(),           // 形状长度
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16  // 数据类型
    );


//    Ort::Value inputTensor_latent = Ort::Value::CreateTensor<Ort::Float16_t>(
//            allocator_info,
//            latents_fp16.data(),
//            latents_fp16.size(),
//            input_node_dims.data(),
//            input_node_dims.size()
//    );

    // make time step input
    // just for test



    std::vector<int64_t> input_node_dims1 = {1};
//    std::vector<float> time_step = {952.0};

    // 转换 time_step 到 FP16
//    std::vector<Ort::Float16_t> time_step_fp16 = {floatToFloat16(952.0f)};
    std::vector<float> time_step = {952.0f};
    size_t time_step_size_in_bytes = time_step.size() * sizeof(float);

    Ort::Value inputTensor_timestep = Ort::Value::CreateTensor(
            allocator_info,
            time_step.data(),                   // 数据指针
            time_step_size_in_bytes,            // 数据大小（字节）
            input_node_dims1.data(),           // 形状指针
            input_node_dims1.size(),           // 形状长度
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16  // 数据类型
    );


//    Ort::Value inputTensor_timestep = Ort::Value::CreateTensor<Ort::Float16_t>(
//            allocator_info,
//            time_step_fp16.data(),
//            time_step_fp16.size(),
//            input_node_dims1.data(),
//            input_node_dims1.size()
//    );


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


// 转换 combined_embedding 到 Float16_t
    std::vector<Ort::Float16_t> combined_embedding_fp16(combined_embedding.size());
    for (size_t i = 0; i < combined_embedding.size(); ++i) {
        combined_embedding_fp16[i] = floatToFloat16(combined_embedding[i]);
    }


    std::vector<int64_t> input_node_dims2 = {2,77,768};
    size_t embedding_size = combined_embedding.size() * sizeof(float );
    Ort::Value inputTensor_embedding = Ort::Value::CreateTensor(
            allocator_info,
            combined_embedding.data(),                   // 数据指针
            embedding_size,            // 数据大小（字节）
            input_node_dims2.data(),           // 形状指针
            input_node_dims2.size(),           // 形状长度
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16  // 数据类型
    );

//    Ort::Value inputTensor_embedding = Ort::Value::CreateTensor<Ort::Float16_t>(
//            allocator_info,
//            combined_embedding_fp16.data(),
//            combined_embedding_fp16.size(),
//            input_node_dims2.data(),
//            input_node_dims2.size()
//    );




    std::vector<Ort::Value> inputTensors;
    // 使用std::move 来移动Ort的对象 因为这些对象只能被移动不能被复制
    inputTensors.emplace_back(std::move(inputTensor_latent));
    inputTensors.emplace_back(std::move(inputTensor_timestep));
    inputTensors.emplace_back(std::move(inputTensor_embedding));


//    std::vector<Ort::Value> outputTensors = ort_session->Run(Ort::RunOptions{nullptr}, inputNodeNames, inputTensors.data(), inputTensors.size(), outputNodeNames, outputNodeCount);
    Ort::RunOptions runOptions;

//    std::vector<Value> Run(const RunOptions& run_options, const char* const* input_names, const Value* input_values, size_t input_count,
//                           const char* const* output_names, size_t output_count);

    // run inference
    std::vector<Ort::Value> outputTensors = ort_session->Run(
            runOptions,
            input_node_names.data(),
            inputTensors.data(),
            inputTensors.size(),
            output_node_names.data(),
            output_node_names.size()
    );


    const Ort::Float16_t *text_feature_ptr = outputTensors[0].GetTensorMutableData<Ort::Float16_t>();
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
    std::vector<Ort::Float16_t> output_test;
    for (int i=0; i < len ; ++i)
        output_test.emplace_back(text_feature_ptr[i]);

    std::vector<float> output_test_float;
    for (int i = 0 ; i < len ; ++i)
    {
        auto temp_value = float16ToFloat(output_test[i]);
        output_test_float.push_back(temp_value);
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