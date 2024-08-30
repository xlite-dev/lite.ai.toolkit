//
// Created by root on 8/28/24.
//

#include "trt_vae.h"
using trtsd::TRTVae;

TRTVae::TRTVae(const std::string &_engine_path) {
    trt_model_path = _engine_path.c_str();
    std::ifstream file(trt_model_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open engine file: " << trt_model_path << std::endl;
        return;
    }
    file.seekg(0, std::ifstream::end);
    size_t model_size = file.tellg();
    file.seekg(0, std::ifstream::beg);

    std::vector<char> model_data(model_size);
    file.read(model_data.data(), model_size);
    file.close();

    trt_runtime.reset(nvinfer1::createInferRuntime(trt_logger));
    trt_engine.reset(trt_runtime->deserializeCudaEngine(model_data.data(), model_size));

    if (!trt_engine) {
        std::cerr << "Failed to deserialize the TensorRT engine." << std::endl;
        return;
    }
    trt_context.reset(trt_engine->createExecutionContext());
    if (!trt_context) {
        std::cerr << "Failed to create execution context." << std::endl;
        return;
    }
    cudaStreamCreate(&stream);
    // make the flexible one input and multi output
    int num_io_tensors = trt_engine->getNbIOTensors(); // get the input and output's num
    buffers.resize(num_io_tensors);
}

TRTVae::~TRTVae() {
    for (auto &buffer : buffers) {
        cudaFree(buffer);
    }
    cudaStreamDestroy(stream);
}

std::vector<float> trt_load_from_bin1(const std::string& filename) {
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

void TRTVae::trt_save_vector_as_image(const std::vector<float>& output_vector, int height, int width, const std::string& filename) {
    // 确保 output_vector 的大小与图像大小匹配
    if (output_vector.size() != height * width * 3) {
        std::cerr << "Vector size does not match image dimensions!" << std::endl;
        return;
    }

    // 重新组织数据，将channels, height, width格式转换为height, width, channels格式
    std::vector<float> reorganized_output(height * width * 3);

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                reorganized_output[h * width * 3 + w * 3 + c] = output_vector[c * height * width + h * width + w];
            }
        }
    }

    // 将重新组织后的数据复制到 cv::Mat
    cv::Mat image(height, width, CV_32FC3, reorganized_output.data());

    // 按照 Python 代码中的逻辑处理数据
    image = (image + 1.0f) * 255.0f / 2.0f;

    // 将值限定在 0-255 的范围内，并转换为 8 位无符号整数类型
    image = cv::min(cv::max(image, 0.0f), 255.0f);

    cv::Mat image_8bit;
    image.convertTo(image_8bit, CV_8UC3);

    cv::cvtColor(image_8bit, image_8bit, cv::COLOR_RGB2BGR);

    // 保存图像
    cv::imwrite(filename, image_8bit);
}


void TRTVae::inference() {

    auto start = std::chrono::high_resolution_clock::now();
    // 先读取之前的latent进行 测试接口这个先留着
    std::string filename = "/home/lite.ai.toolkit/trt_final_latent_data.bin";
    std::vector<float> latent = trtcv::utils::transform::trt_load_from_bin(filename);
    std::vector<float> latent_input(latent.size(),0);

    std::transform(latent.begin(),latent.end(),latent_input.begin(),[](float x){return (1.0f / 0.18215 * x);});

    // 分配cuda内存
    cudaMalloc(&buffers[0], latent_input.size() * sizeof(float));
    trt_context->setTensorAddress(input_names,buffers[0]);

    // output buffer
    cudaMalloc(&buffers[1],output_size * sizeof (half));
    trt_context->setTensorAddress(output_names,buffers[1]);

    cudaMemcpyAsync(buffers[0], latent_input.data(), latent_input.size() * sizeof(float ),
                    cudaMemcpyHostToDevice, stream);
    nvinfer1::Dims inputDims;
    inputDims.nbDims = static_cast<int32_t>(input_node_dims.size()); // 确保 nbDims 正确设置

    // set trt input dims
    std::transform(input_node_dims.begin(),input_node_dims.end(),inputDims.d,[](int64_t dim){return static_cast<int32_t>(dim);});
    trt_context->setInputShape("latent", inputDims);
    bool status = trt_context->enqueueV3(stream);

    if (!status){
        std::cerr << "Failed to infer by TensorRT." << std::endl;
        return;
    }

    std::vector<half> output_trt_half(output_size);
    cudaMemcpyAsync(output_trt_half.data(), buffers[1], output_size * sizeof(half),
                    cudaMemcpyDeviceToHost, stream);

    // float output vector
    std::vector<float> output_trt_float(output_size, 0);
    // half to float
    std::transform(output_trt_half.begin(),output_trt_half.end(),
                   output_trt_float.begin(),[](half h){return __half2float(h);});

    TRTVae::trt_save_vector_as_image(output_trt_float, output_node_dims[2], output_node_dims[3],
                             "/home/lite.ai.toolkit/trt_result.png");

    std::cout<<"trt vae inference done!"<<std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "没有加载engine的耗时是: " << duration << " 毫秒" << std::endl;

}

