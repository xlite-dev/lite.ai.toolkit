//
// Created by root on 11/15/24.
//

#include "trt_face_restoration_mt.h"
#include "fstream"

trt_face_restoration_mt::trt_face_restoration_mt(std::string &model_path) {
    std::ifstream file(model_path, std::ios::binary);

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

    trt_runtime.reset(nvinfer1::createInferRuntime(logger));
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


void trt_face_restoration_mt::detect(cv::Mat &face_swap_image, std::vector<cv::Point2f> &target_landmarks_5,
                                     const std::string &face_enchaner_path) {
    auto ori_image = face_swap_image.clone();

    cv::Mat crop_image;
    cv::Mat affine_matrix;
    std::tie(crop_image,affine_matrix) = face_utils::warp_face_by_face_landmark_5(face_swap_image,target_landmarks_5,face_utils::FFHQ_512);

    std::vector<float> crop_size = {512,512};
    cv::Mat box_mask = face_utils::create_static_box_mask(crop_size);
    std::vector<cv::Mat> crop_mask_list;
    crop_mask_list.emplace_back(box_mask);

    cv::cvtColor(crop_image,crop_image,cv::COLOR_BGR2RGB);
    crop_image.convertTo(crop_image,CV_32FC3,1.f / 255.f);
    crop_image.convertTo(crop_image,CV_32FC3,2.0f,-1.f);

    std::vector<float> input_vector;
    trtcv::utils::transform::create_tensor(crop_image,input_vector,input_node_dims,trtcv::utils::transform::CHW);

    // 拷贝
    cudaMemcpyAsync(buffers[0],input_vector.data(),1 * 3 * 512 * 512 * sizeof(float),cudaMemcpyHostToDevice,stream);

    // 同步
    cudaStreamSynchronize(stream);

    // 推理
    bool status = trt_context->enqueueV3(stream);
    if (!status) {
        std::cerr << "Failed to inference" << std::endl;
        return;
    }


    // 同步
    cudaStreamSynchronize(stream);

    // 获取输出
    std::vector<float> output_vector(1 * 3 * 512 * 512);
    cudaMemcpyAsync(output_vector.data(),buffers[1],1 * 3 * 512 * 512 * sizeof(float),cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);

    // 后处理
    int channel = 3;
    int height = 512;
    int width = 512;
    std::vector<float> output(channel * height * width);
    output.assign(output_vector.begin(),output_vector.end());

    std::transform(output.begin(),output.end(),output.begin(),
                   [](double x){return std::max(-1.0,std::max(-1.0,std::min(1.0,x)));});

    std::transform(output.begin(),output.end(),output.begin(),
                   [](double x){return (x + 1.f) /2.f;});


    std::vector<float> transposed_data(channel * height * width);
    for (int c = 0; c < channel; ++c){
        for (int h = 0 ; h < height; ++h){
            for (int w = 0; w < width ; ++w){
                int src_index = c * (height * width) + h * width + w;
                int dst_index = h * (width * channel) + w *  channel + c;
                transposed_data[dst_index] = output[src_index];
            }
        }
    }

    std::transform(transposed_data.begin(),transposed_data.end(),transposed_data.begin(),
                   [](float x){return std::round(x * 255.f);});

    std::transform(transposed_data.begin(), transposed_data.end(), transposed_data.begin(),
                   [](float x) { return static_cast<uint8_t>(x); });


    cv::Mat mat(height, width, CV_32FC3, transposed_data.data());
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);


    auto crop_mask = crop_mask_list[0];
    cv::Mat paste_frame = face_utils::paste_back(ori_image,mat,crop_mask,affine_matrix);

    cv::Mat dst_image = face_utils::blend_frame(ori_image,paste_frame);

    cv::imwrite(face_enchaner_path,dst_image);

}


trt_face_restoration_mt::~trt_face_restoration_mt() {
    // don't need free by manunly
    for (auto buffer : buffers) {
        cudaFree(buffer);
    }
    trt_context.reset();
    cudaStreamDestroy(stream);
}

