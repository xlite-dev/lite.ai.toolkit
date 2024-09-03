#include "unet.h"


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



void save_output_as_image(const std::vector<float>& output, const std::string& filename) {
    if (output.empty()) {
        std::cerr << "Empty output" << std::endl;
        return;
    }

    int channels = 4;
    int height = 64;
    int width = 64;


    if (height * width * channels != output.size()) {
        std::cerr << "Incorrect dimensions" << std::endl;
        return;
    }

    cv::Mat image(height, width, CV_32FC4);

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
                image.at<cv::Vec4f>(h, w)[c] = output[(c * height + h) * width + w];
            }
        }
    }

    cv::normalize(image, image, 0, 255, cv::NORM_MINMAX);

    image.convertTo(image, CV_8UC4);

    cv::imwrite(filename, image);
}


std::vector<float> read_latent_data(const std::string& filename, int size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return {};
    }
    std::vector<float> latents(size);
    file.read(reinterpret_cast<char*>(latents.data()), size * sizeof(float));
    return latents;
}


void save_to_bin(const std::vector<float>& data, const std::string& filename) {
    std::ofstream outfile(filename, std::ios::out | std::ios::binary);
    if (outfile.is_open()) {
        outfile.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
        outfile.close();
    } else {
        std::cerr << "Failed to open file: " << filename << std::endl;
    }
}

std::vector<float> trt_load_from_bin2(const std::string& filename) {
    std::ifstream infile(filename, std::ios::in | std::ios::binary);
    std::vector<float> data;

    if (infile.is_open()) {
        infile.seekg(0, std::ios::end);
        size_t size = infile.tellg();
        infile.seekg(0, std::ios::beg);

        data.resize(size / sizeof(float));
        infile.read(reinterpret_cast<char*>(data.data()), size);
        infile.close();
    } else {
        std::cerr << "Failed to open file: " << filename << std::endl;
    }

    return data;
}



void UNet::inference(std::vector<std::vector<float>> clip_output, std::vector<float> &latent_final_output,std::string scheduler_config_path) {

    auto scheduler = Scheduler::DDIMScheduler(scheduler_config_path);
    scheduler.set_timesteps(30);
    std::vector<int> timesteps;
    scheduler.get_timesteps(timesteps);
    auto init_noise_sigma = scheduler.get_init_noise_sigma();
    std::vector<float> latents(1 * 4 * 64 * 64);
    generate_latents(latents, 1, 4, 64, 64, init_noise_sigma);
    latents.insert(latents.end(), latents.begin(), latents.end());

    for (auto t : timesteps)
    {
        std::vector<float> output_embedding = clip_output[0];
        std::vector<float> negative_output_embedding = clip_output[1];

        size_t total_size = output_embedding.size() + negative_output_embedding.size();
        std::vector<float> combined_embedding(total_size);
        std::copy(output_embedding.begin(), output_embedding.end(), combined_embedding.begin());
        std::copy(negative_output_embedding.begin(), negative_output_embedding.end(), combined_embedding.begin() + output_embedding.size());

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

        float t_float = static_cast<float>(t);

        std::vector<float> time_step = {t_float};
        std::vector<Ort::Float16_t> time_step_fp16(time_step.size());
        for (size_t i = 0; i < time_step.size(); ++i) {
            time_step_fp16[i] = Ort::Float16_t(time_step[i]);
        }
        std::vector<int64_t> input_node_dims1 = {1};

        Ort::Value inputTensor_timestep = Ort::Value::CreateTensor<Ort::Float16_t>(
                memory_info,
                time_step_fp16.data(),
                time_step_fp16.size(),
                input_node_dims1.data(),
                input_node_dims1.size()
        );


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

        std::vector<std::vector<float>> output;
        output.resize(batch);
        for (int i = 0; i < batch; ++i) {
            output[i].resize(channels * height * width);
            for (int j = 0; j < channels * height * width; ++j) {
                output[i][j] = static_cast<float>(noise_preds[i * channels * height * width + j]);
            }
        }

        std::vector<float> noise_pred_uncond(1 * 4 * 64 * 64, 0 );
        std::vector<float> noise_pred_text(1  * 4 * 64 * 64 , 0);

        std::copy(output[0].begin(), output[0].begin() + 1 * 4 * 64 * 64, noise_pred_text.begin());
        std::copy(output[1].begin() , output[1].begin()+ 1 * 4 * 64 * 64, noise_pred_uncond.begin());

        std::vector<float> noise_pred(1 * 4 * 64 * 64,0);
        for (size_t i = 0; i < noise_pred.size(); ++i)
        {
            noise_pred[i] = noise_pred_uncond[i] + 7.5f * (noise_pred_text[i] - noise_pred_uncond[i]);
        }


        std::vector<float> pred_sample;
        scheduler.step(noise_pred, {1, 4, 64, 64}, latents, {1, 4, 64, 64}, pred_sample, t);
        latents.clear();
        latents.assign(pred_sample.begin(), pred_sample.end());

        latents.insert(latents.end(), latents.begin(), latents.end());

    }

    latent_final_output.assign(latents.begin(), latents.end());

}