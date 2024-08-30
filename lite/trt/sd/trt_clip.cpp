//
// Created by wangzijian on 8/7/24.
//

#include "trt_clip.h"
#include "tokenizer.h"
#include "vocab.h"
using trtsd::TRTClip;
using namespace trt_tokenizer;

TRTClip::~TRTClip() {
    // don't need free by manunly
    for (auto buffer : buffers) {
        cudaFree(buffer);
    }
    cudaStreamDestroy(stream);
}

TRTClip::TRTClip(const std::string &engine_path) {
    trt_model_path = engine_path.c_str();
    std::ifstream file(trt_model_path, std::ios::binary);

    if (!file.good()) {
        std::cerr << "Failed to read model file: " << trt_model_path << std::endl;
        return;
    }

    file.seekg(0, std::ifstream::end);
    size_t model_size = file.tellg();
    file.seekg(0, std::ifstream::beg);
    std::vector<char> model_data(model_size);
    file.read(model_data.data(), model_size);
    file.close();

    trt_runtime.reset(nvinfer1::createInferRuntime(trt_logger));
    // engine deserialize
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

void TRTClip::encode_text(std::vector<std::string> input_text, std::vector<std::vector<int>> &output) {

    CLIPTokenizer tokenizer(VERSION_1_x);
    std::string str(reinterpret_cast<char*>(merges_utf8_c_str),sizeof(merges_utf8_c_str));
    tokenizer.load_from_merges(str);

    auto on_new_token_cb = [](std::string& str, std::vector<int32_t>& tokens) -> bool {
        return false;
    };

    for (int i = 0 ; i < input_text.size(); ++i)
    {
        auto temp = tokenizer.tokenize(input_text[i], on_new_token_cb);
        temp.push_back(end_flag_num);
        if (temp.size() < input_axes) {
            temp.resize(input_axes, 0);
        }
        output.push_back(temp);
    }

}

void TRTClip::inference(std::vector<std::string> input, std::vector<std::vector<float>> &output) {
    std::vector<std::vector<int>> output_encode;

    encode_text(input,output_encode);

    // flat out output_encode
    std::vector<int32_t> flat_output_encode;
    for (const auto& vec : output_encode) {
        flat_output_encode.insert(flat_output_encode.end(), vec.begin(), vec.end());
    }

    // get the input's name and malloc
    // input buffer
    auto batch = output_encode.size();
    cudaMalloc(&buffers[0],flat_output_encode.size() * sizeof (int32_t));
    trt_context->setTensorAddress(input_names,buffers[0]);

    // output buffer
    cudaMalloc(&buffers[1],batch * output_tensor_size * sizeof (float));
    trt_context->setTensorAddress(output_names,buffers[1]);

    cudaMalloc(&buffers[2], output_tensor_size * sizeof (float));
    trt_context->setTensorAddress("2233",buffers[2]);

    std::vector<int> input_dims = { static_cast<int>(batch), input_axes};

    // infer
    cudaMemcpyAsync(buffers[0], flat_output_encode.data(), flat_output_encode.size() * sizeof(int32_t ),
                    cudaMemcpyHostToDevice, stream);
    // set input shape for dynamic shape
    nvinfer1::Dims inputDims;
    inputDims.nbDims = 2;
    inputDims.d[0] = batch;
    inputDims.d[1] = input_axes;
    // input name should be equal
    trt_context->setInputShape("input_ids", inputDims);

    bool status = trt_context->enqueueV3(stream);

    if (!status){
        std::cerr << "Failed to infer by TensorRT." << std::endl;
        return;
    }

    std::vector<float> output_trt(batch * output_tensor_size);

    cudaMemcpyAsync(output_trt.data(), buffers[1], batch * output_tensor_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

    for (int i = 0 ; i < batch ; ++i)
    {
        std::vector<float> temp;
        for (int j = 0 ; j < output_tensor_size ; ++j)
        {
            temp.push_back(output_trt[ i * output_tensor_size + j]);
        }
        output.push_back(temp);
        temp.clear();
    }

    std::cout<<"trt clip inference done!"<<std::endl;

}