//
// Created by root on 3/27/25.
//

#include "trt_vae_encoder.h"
using trtsd::TRTVaeEncoder;

TRTVaeEncoder::TRTVaeEncoder(const std::string &_engine_path) {
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

TRTVaeEncoder::~TRTVaeEncoder()  {
    for (auto &buffer : buffers) {
        cudaFree(buffer);
    }
    cudaStreamDestroy(stream);
}

void TRTVaeEncoder::inference(const std::vector<float> &input_images, std::vector<float> &output_latents) {
    std::vector<half > vae_encoder_input(input_images.size(), __float2half(0));
    std::transform(input_images.begin(),input_images.end(),vae_encoder_input.begin(),
                   [](float x){return __float2half(x);});

    cudaMalloc(&buffers[0], vae_encoder_input.size() * sizeof(half));
    trt_context->setTensorAddress(input_names,buffers[0]);

    cudaMalloc(&buffers[1],output_size * sizeof (half));
    trt_context->setTensorAddress(output_names,buffers[1]);

    cudaMemcpyAsync(buffers[0], vae_encoder_input.data(), vae_encoder_input.size() * sizeof(half),
                    cudaMemcpyHostToDevice, stream);
    nvinfer1::Dims inputDims;
    inputDims.nbDims = static_cast<int32_t>(input_node_dims.size()); // 确保 nbDims 正确设置

    // 设置正确的输入维度
    std::transform(input_node_dims.begin(),input_node_dims.end(),inputDims.d,[](int64_t dim){return static_cast<int32_t>(dim);});
    trt_context->setInputShape("images", inputDims);
    bool status = trt_context->enqueueV3(stream);

    if (!status){
        std::cerr << "Failed to infer by TensorRT." << std::endl;
        return;
    }

    std::vector<half> output_trt_half(output_size);
    output_latents.resize(output_size);

    cudaMemcpyAsync(output_trt_half.data(), buffers[1], output_size * sizeof(half),
                    cudaMemcpyDeviceToHost, stream);



    std::transform(output_trt_half.begin(),output_trt_half.end(),
                   output_latents.begin(),[](half h){return (__half2float(h)) * 0.1825;});


    std::cout<<"trt vae inference done!"<<std::endl;

}