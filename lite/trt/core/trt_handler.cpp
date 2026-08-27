//
// Created by wangzijian on 24-7-11.
//

#include "trt_handler.h"


using trtcore::BasicTRTHandler; // using namespace

BasicTRTHandler::BasicTRTHandler(const std::string &_trt_model_path, unsigned int _num_threads) : log_id(_trt_model_path.data()),num_threads(_num_threads){
    trt_model_path = _trt_model_path.data(); // model path
    initialize_handler();
    print_debug_string();
}

BasicTRTHandler::~BasicTRTHandler() {
    // don't need free by manunly
    for (auto buffer : buffers) {
        if (buffer) cudaFree(buffer);
    }
    if (stream) cudaStreamDestroy(stream);
}


void BasicTRTHandler::initialize_handler() {
    // read engine file
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

    for (int i = 0; i < num_io_tensors; ++i) {
        auto tensor_name = trt_engine->getIOTensorName(i);
        nvinfer1::Dims tensor_dims = trt_engine->getTensorShape(tensor_name);

        // input
        if (i==0)
        {
            size_t tensor_size = 1;
            for (int j = 0; j < tensor_dims.nbDims; ++j) {
                tensor_size *= tensor_dims.d[j];
                input_node_dims.push_back(tensor_dims.d[j]);
            }
            cudaMalloc(&buffers[i], tensor_size * sizeof(float));
            trt_context->setTensorAddress(tensor_name, buffers[i]);
            continue;
        }

        // output
        size_t tensor_size = 1;

        std::vector<int64_t> output_node;
        for (int j = 0; j < tensor_dims.nbDims; ++j) {
            output_node.push_back(tensor_dims.d[j]);
            tensor_size *= tensor_dims.d[j];
        }
        output_node_dims.push_back(output_node);

        cudaMalloc(&buffers[i], tensor_size * sizeof(float));
        trt_context->setTensorAddress(tensor_name, buffers[i]);
        output_tensor_size++;
    }


}

void BasicTRTHandler::print_debug_string() {
    std::cout << "TensorRT model loaded from: " << trt_model_path << std::endl;
    std::cout << "Input tensor size: " << input_tensor_size << std::endl;
    std::cout << "Output tensor size: " << output_tensor_size << std::endl;
}

