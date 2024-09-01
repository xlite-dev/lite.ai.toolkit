//
// Created by wangzijian on 8/27/24.
//

#include "vae.h"
using ortsd::Vae;


Vae::Vae(const std::string &_onnx_path, unsigned int _num_threads) :
        log_id(_onnx_path.data()), num_threads(_num_threads) {
    onnx_path = _onnx_path.data();
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(num_threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_options.SetLogSeverityLevel(4);
    ort_session = new Ort::Session(ort_env, onnx_path, session_options);
    std::cout << "Load " << onnx_path << " done!" << std::endl;
}

Vae::~Vae() {
    delete ort_session;
}

std::vector<float> load_from_bin(const std::string& filename) {
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

void save_vector_as_image(const std::vector<float>& output_vector, int height, int width, const std::string& filename) {

    if (output_vector.size() != height * width * 3) {
        std::cerr << "Vector size does not match image dimensions!" << std::endl;
        return;
    }

    std::vector<float> reorganized_output(height * width * 3);

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                reorganized_output[h * width * 3 + w * 3 + c] = output_vector[c * height * width + h * width + w];
            }
        }
    }


    cv::Mat image(height, width, CV_32FC3, reorganized_output.data());

    image = (image + 1.0f) * 255.0f / 2.0f;

    image = cv::min(cv::max(image, 0.0f), 255.0f);

    cv::Mat image_8bit;
    image.convertTo(image_8bit, CV_8UC3);

    cv::cvtColor(image_8bit, image_8bit, cv::COLOR_RGB2BGR);

    cv::imwrite(filename, image_8bit);
}


void Vae::inference(const std::vector<float> &unet_input, const std::string save_path) {

    std::vector<float> latent_input(unet_input.size(),0);

    for (int i = 0; i < unet_input.size(); i++)
    {
        latent_input[i] = 1.0f / 0.18215 * unet_input[i];
    }


    std::vector<int64_t> input_node_dims = {1, 4, 64, 64};


    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memory_info,
            latent_input.data(),
            latent_input.size(),
            input_node_dims.data(),
            input_node_dims.size()
    );

    Ort::RunOptions runOptions;

    // run inference
    std::vector<Ort::Value> ort_outputs = ort_session->Run(
            runOptions,
            input_node_names.data(),
            &inputTensor,
            1,
            output_node_names.data(),
            output_node_names.size()
    );

    const Ort::Float16_t* vae_preds = ort_outputs[0].GetTensorData<Ort::Float16_t>();


    auto shape_info = ort_outputs[0].GetTensorTypeAndShapeInfo();
    auto dims = shape_info.GetShape();

    int batch = dims[0];
    int channels = dims[1];
    int height = dims[2];
    int width = dims[3];

    std::vector<float>  output1;
    output1.resize(1 * channels * height * width);

    for (size_t i = 0; i < output1.size(); ++i)
    {
        output1[i] = static_cast<float>(vae_preds[i]);
    }

    save_vector_as_image(output1, 512, 512, save_path);

}

