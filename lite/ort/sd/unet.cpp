#include "unet.h"
#include <random>
#include <cstdint>
#include <iomanip>
#include "tokenizer.h"
#include "onnxruntime_cxx_api.h"

using ortsd::UNet;

void generate_latents(std::vector<float>& latents, int batch_size, int unet_channels, int latent_height, int latent_width, float init_noise_sigma) {
    size_t total_size = batch_size * unet_channels * latent_height * latent_width;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < total_size; ++i) {
        latents[i] = dist(gen) * init_noise_sigma;
    }
}

UNet::UNet(const std::string &_onnx_path, unsigned int _num_threads) :
        log_id(_onnx_path.data()), num_threads(_num_threads) {
    onnx_path = _onnx_path.data();
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(num_threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_options.SetLogSeverityLevel(4);
    ort_session = new Ort::Session(ort_env, onnx_path, session_options);
    std::cout << "Load " << onnx_path << " done!" << std::endl;
}

UNet::~UNet() {
    delete ort_session;
}



void save_output_as_image(const std::vector<std::vector<float>>& output, const std::string& filename) {
    if (output.empty() || output[0].empty()) {
        std::cerr << "Empty output" << std::endl;
        return;
    }

    int batch = output.size();
    int channels = 4;  // UNet输出通常是4通道
    int height = 64;   // 假设高度是64
    int width = 64;    // 假设宽度是64

    // 检查维度是否正确
    if (height * width * channels != output[0].size()) {
        std::cerr << "Incorrect dimensions" << std::endl;
        return;
    }

    // 我们只处理第一个batch的数据
    cv::Mat image(height, width, CV_32FC4);

    // 重新排列数据
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
                image.at<cv::Vec4f>(h, w)[c] = output[0][(c * height + h) * width + w];
            }
        }
    }

    // 归一化到0-255范围
    cv::normalize(image, image, 0, 255, cv::NORM_MINMAX);

    // 转换为8位无符号整型
    image.convertTo(image, CV_8UC4);

    // 保存图像
    cv::imwrite(filename, image);
}


void UNet::inference(std::vector<std::string> input, std::vector<std::vector<float>> &output) {
    auto scheduler = Scheduler::DDIMScheduler("/home/lite.ai.toolkit/lite/ort/sd/scheduler_config.json");
    scheduler.set_timesteps(30);
    std::vector<int> timesteps;
    scheduler.get_timesteps(timesteps);
    auto init_noise_sigma = scheduler.get_init_noise_sigma();

    std::vector<float> latents(1 * 4 * 64 * 64);
    generate_latents(latents, 1, 4, 64, 64, init_noise_sigma);
    latents.insert(latents.end(), latents.begin(), latents.end());

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Convert latents to FP16
    std::vector<Ort::Float16_t> latents_fp16(latents.size());
    for (size_t i = 0; i < latents.size(); ++i) {
        latents_fp16[i] = Ort::Float16_t(latents[i]);
    }

    std::vector<int64_t> input_node_dims = {2, 4, 64, 64};
    Ort::Value inputTensor_latent = Ort::Value::CreateTensor<Ort::Float16_t>(
            memory_info,
            latents_fp16.data(),
            latents_fp16.size(),
            input_node_dims.data(),
            input_node_dims.size()
    );

    std::vector<int64_t> input_node_dims1 = {1};
    std::vector<float> time_step = {952.0f};
    std::vector<Ort::Float16_t> time_step_fp16(time_step.size());
    for (size_t i = 0; i < time_step.size(); ++i) {
        time_step_fp16[i] = Ort::Float16_t(time_step[i]);
    }

    Ort::Value inputTensor_timestep = Ort::Value::CreateTensor<Ort::Float16_t>(
            memory_info,
            time_step_fp16.data(),
            time_step_fp16.size(),
            input_node_dims1.data(),
            input_node_dims1.size()
    );

    std::string onnx_path = "/home/lite.ai.toolkit/examples/hub/onnx/sd/clip_model.onnx";
    Clip clip(onnx_path);

    std::vector<std::string> input_text = {"a beauty girl"};
    std::vector<std::string> negative_input_text = {"anime"};
    std::vector<std::vector<float>> output_embedding;
    std::vector<std::vector<float>> negative_output_embedding;

    clip.inference(input_text, output_embedding);
    clip.inference(negative_input_text, negative_output_embedding);

    size_t total_size = output_embedding.size() * output_embedding[0].size() +
                        negative_output_embedding.size() * negative_output_embedding[0].size();
    std::vector<float> combined_embedding(total_size);
    std::copy(output_embedding[0].begin(), output_embedding[0].end(), combined_embedding.begin());
    std::copy(negative_output_embedding[0].begin(), negative_output_embedding[0].end(), combined_embedding.begin() + output_embedding[0].size());

    std::vector<int64_t> input_node_dims2 = {2, 77, 768};

    std::vector<Ort::Float16_t> combined_embedding_fp16(combined_embedding.size());
    for (size_t i = 0; i < combined_embedding.size(); ++i) {
        combined_embedding_fp16[i] = Ort::Float16_t(combined_embedding[i]);
    }

    Ort::Value inputTensor_embedding = Ort::Value::CreateTensor<Ort::Float16_t>(
            memory_info,
            combined_embedding_fp16.data(),
            combined_embedding_fp16.size(),
            input_node_dims2.data(),
            input_node_dims2.size()
    );

    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(std::move(inputTensor_latent));
    inputTensors.push_back(std::move(inputTensor_timestep));
    inputTensors.push_back(std::move(inputTensor_embedding));

    Ort::RunOptions runOptions;

    std::vector<Ort::Value> outputTensors = ort_session->Run(
            runOptions,
            input_node_names.data(),
            inputTensors.data(),
            inputTensors.size(),
            output_node_names.data(),
            output_node_names.size()
    );

    const Ort::Float16_t* noise_preds = outputTensors[0].GetTensorData<Ort::Float16_t>();
    auto shape_info = outputTensors[0].GetTensorTypeAndShapeInfo();
    auto dims = shape_info.GetShape();

    int batch = dims[0];
    int channels = dims[1];
    int height = dims[2];
    int width = dims[3];

    output.resize(batch);
    for (int i = 0; i < batch; ++i) {
        output[i].resize(channels * height * width);
        for (int j = 0; j < channels * height * width; ++j) {
            output[i][j] = static_cast<float>(noise_preds[i * channels * height * width + j]);
        }
    }

    std::vector<float> noise_pred_uncond;
    noise_pred_uncond.resize(1 * 4 * 64 * 64);

    std::vector<float> noise_pred_text;
    noise_pred_text.resize(1  * 4 * 64 * 64);

    // 将output的前1 * 4 * 64 * 64个元素复制到noise_pred_uncond
    std::copy(output[0].begin(), output[0].begin() + 1 * 4 * 64 * 64, noise_pred_uncond.begin());
    // 将output的后1 * 4 * 64 * 64个元素复制到noise_pred_text
    std::copy(output[0].begin() + 1 * 4 * 64 * 64, output[0].end(), noise_pred_text.begin());

    std::vector<float> noise_pred;
    noise_pred.resize(1 * 4 * 64 * 64);
    for (size_t i = 0; i < noise_pred.size(); ++i)
    {
        noise_pred[i] = noise_pred_uncond[i] + 7.0 * (noise_pred_text[i] - noise_pred_uncond[i]);
        noise_pred[i] = static_cast<float>(noise_preds[i]);
    }

    std::cout<<"output size: "<<noise_pred.size()<<std::endl;

    save_output_as_image(output, "/home/lite.ai.toolkit/output_image.png");

}