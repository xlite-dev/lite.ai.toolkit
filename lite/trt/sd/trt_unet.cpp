//
// Created by root on 8/28/24.
//

#include "trt_unet.h"
using trtsd::TRTUNet;

TRTUNet::TRTUNet(const std::string &_engine_path) {
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

TRTUNet::~TRTUNet() {
    for (auto &buffer : buffers) {
        cudaFree(buffer);
    }
    cudaStreamDestroy(stream);
}

std::vector<float> TRTUNet::convertToFloat(const std::vector<half> &half_vec) {
    std::vector<float> float_vec(half_vec.size());
    std::transform(half_vec.begin(), half_vec.end(), float_vec.begin(),
                   [](half h) { return __half2float(h); });
    return float_vec;
}



std::vector<float> trt_load_from_bin(const std::string& filename) {
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

void TRTUNet::get_timesteps(const std::vector<int> &time_steps, std::vector<int>& new_time_steps, int &num_inference_steps,float strength) {
    auto init_timestep = std::min(int(num_inference_steps * strength), num_inference_steps);
    auto t_start = std::max(num_inference_steps - init_timestep, 0);
    // 从 t_start 开始截取时间步
    new_time_steps = std::vector<int>(time_steps.begin() + t_start, time_steps.end());
    num_inference_steps = 30 - t_start;
}

void TRTUNet::inference(const std::vector<std::vector<float>> &clip_output, std::vector<float> &unet_output,std::string scheduler_config_path) {


    auto scheduler = Scheduler::DDIMScheduler(scheduler_config_path);
    scheduler.set_timesteps(30);
    std::vector<int> timesteps;
    scheduler.get_timesteps(timesteps);
    auto init_noise_sigma = scheduler.get_init_noise_sigma();
    std::vector<float> latents(final_latent_outputsize);
    trtcv::utils::transform::trt_generate_latents(latents, 1, 4, 64, 64, init_noise_sigma);
    latents.insert(latents.end(), latents.begin(), latents.end());

    std::vector<float> output_embedding = clip_output[0];
    std::vector<float> negative_output_embedding = clip_output[1];

    size_t total_size = output_embedding.size()  + negative_output_embedding.size();
    std::vector<float> combined_embedding(total_size);
    std::copy(output_embedding.begin(), output_embedding.end(), combined_embedding.begin());
    std::copy(negative_output_embedding.begin(), negative_output_embedding.end(), combined_embedding.begin() + output_embedding.size());

    std::vector<half> combined_embedding_fp16(combined_embedding.size());
    std::transform(combined_embedding.begin(), combined_embedding.end(), combined_embedding_fp16.begin(),[](float f) { return __float2half(f);});


    std::vector<half> latents_fp16(latents.size(),0);
    std::transform(latents.begin(), latents.end(), latents_fp16.begin(),[](float f) { return __float2half(f);});


    cudaMalloc(&buffers[2], combined_embedding_fp16.size() * sizeof(half));
    trt_context->setTensorAddress(input_names[2],buffers[2]);
    cudaMemcpyAsync(buffers[2], combined_embedding_fp16.data(), combined_embedding_fp16.size() * sizeof(half ),
                    cudaMemcpyHostToDevice, stream);
    nvinfer1::Dims hiddenStateDims;
    hiddenStateDims.nbDims = 3;
    hiddenStateDims.d[0] = input_node_dims_encoder_hidden_states[0];
    hiddenStateDims.d[1] = input_node_dims_encoder_hidden_states[1];
    hiddenStateDims.d[2] = input_node_dims_encoder_hidden_states[2];
    trt_context->setInputShape("encoder_hidden_states", hiddenStateDims);

    cudaMalloc(&buffers[0], latents_fp16.size() * sizeof(half));
    trt_context->setTensorAddress(input_names[0],buffers[0]);
    cudaMemcpyAsync(buffers[0], latents_fp16.data(), latents_fp16.size() * sizeof(half ),
                    cudaMemcpyHostToDevice, stream);
    nvinfer1::Dims sampleDims;
    sampleDims.nbDims = 4;
    sampleDims.d[0] = input_node_dims_sample[0];
    sampleDims.d[1] = input_node_dims_sample[1];
    sampleDims.d[2] = input_node_dims_sample[2];
    sampleDims.d[3] = input_node_dims_sample[3];
    trt_context->setInputShape("sample", sampleDims);

    cudaMalloc(&buffers[3], unet_outputsize * sizeof(half));
    trt_context->setTensorAddress(output_names,buffers[3]);
    std::vector<half> output_trt(unet_outputsize);

    std::vector<float> final_latent_output(final_latent_outputsize,0);
    std::vector<float> pred_sample;

    for (auto t : timesteps)
    {
        float t_f = t;
        std::vector<half> t_fp16(1,__float2half(t_f));
        cudaMalloc(&buffers[1], t_fp16.size() * sizeof(half));
        trt_context->setTensorAddress(input_names[1],buffers[1]);
        cudaMemcpyAsync(buffers[1], t_fp16.data(), t_fp16.size() * sizeof(half ),
                        cudaMemcpyHostToDevice, stream);

        nvinfer1::Dims timeStepDims;
        timeStepDims.nbDims = 1;
        timeStepDims.d[0] = 1;
        trt_context->setInputShape("timestep", timeStepDims);


        bool status = trt_context->enqueueV3(stream);

        if (!status){
            std::cerr << "Failed to infer by TensorRT." << std::endl;
            return;
        }

        cudaMemcpyAsync(output_trt.data(), buffers[3], unet_outputsize * sizeof(half),
                        cudaMemcpyDeviceToHost, stream);

        std::vector<half> noise_pred_uncond(noise_pred_outputsize);
        std::vector<half > noise_pred_text(noise_pred_outputsize);

        std::copy(output_trt.begin(), output_trt.begin() +noise_pred_outputsize, noise_pred_text.begin());
        std::copy(output_trt.begin() + noise_pred_outputsize, output_trt.end(), noise_pred_uncond.begin());


        std::vector<float> noise_pred_uncond_float = convertToFloat(noise_pred_uncond);
        std::vector<float> noise_pred_text_float = convertToFloat(noise_pred_text);


        std::vector<float> noise_pred(noise_pred_outputsize,0);


        float cfgScale = cfg_scale;
        std::transform(noise_pred_uncond_float.begin(),
                       noise_pred_uncond_float.end(),
                       noise_pred_text_float.begin(),
                       noise_pred.begin(),
                       [cfgScale](float uncond, float text) {
                           return uncond + cfgScale * (text - uncond);
                       });

        std::vector<float> latents_fp32 = convertToFloat(latents_fp16);
        scheduler.step(noise_pred,noise_pred_dims, latents_fp32, noise_pred_dims,
                       pred_sample, t);

        std::vector<half> pred_sample_fp16(pred_sample.size(),0);
        std::transform(pred_sample.begin(), pred_sample.end(),
                       pred_sample_fp16.begin(),[](float f) { return __float2half(f);});

        unet_output.assign(pred_sample.begin(),pred_sample.end());

        latents_fp16.clear();
        latents_fp16.assign(pred_sample_fp16.begin(),pred_sample_fp16.end());
        latents_fp16.insert(latents_fp16.end(), latents_fp16.begin(), latents_fp16.end());

        // update buffers value
        cudaMemcpyAsync(buffers[0], latents_fp16.data(), latents_fp16.size() * sizeof(half ),
                        cudaMemcpyHostToDevice, stream);
    }


    std::vector<float> output_trt_float(final_latent_outputsize);
    output_trt_float.assign(final_latent_output.begin(),final_latent_output.end());



}


std::vector<float> load_binary_file(const std::string& filename) {
    // 打开文件
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }

    // 获取文件大小
    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // 创建适当大小的vector
    size_t num_elements = file_size / sizeof(float);
    std::vector<float> data(num_elements);

    // 读取数据
    file.read(reinterpret_cast<char*>(data.data()), file_size);

    // 检查是否读取成功
    if (!file) {
        throw std::runtime_error("只读取了 " + std::to_string(file.gcount()) + " 字节，应为 " + std::to_string(file_size));
    }

    return data;
}




void TRTUNet::inference(const std::vector<std::vector<float>> &clip_output, const std::vector<float> &latent_input,
                        std::vector<float> &unet_output, std::string scheduler_config_path) {

    auto scheduler = Scheduler::DDIMScheduler(scheduler_config_path);
    int num_inference_step = 30;

    scheduler.set_timesteps(num_inference_step);
    std::vector<int> timesteps;
    scheduler.get_timesteps(timesteps);
    std::vector<int> new_time_steps;
    get_timesteps(timesteps, new_time_steps, num_inference_step, 0.75);

    auto init_noise_sigma = scheduler.get_init_noise_sigma();
    std::vector<float> latents(final_latent_outputsize);
    trtcv::utils::transform::trt_generate_latents(latents, 1, 4, 64, 64, init_noise_sigma);

    std::vector<int> latent_input_size = { 1,4,64,64};
    std::vector<float> latent_input_final;

    std::vector<float> latent_input_temp(latent_input.begin(),latent_input.end());
    scheduler.add_noise(latent_input_temp,latent_input_size,latents,latent_input_size,694,latent_input_final);



    latent_input_final.insert(latent_input_final.end(), latent_input_final.begin(), latent_input_final.end());

    std::vector<float> output_embedding = clip_output[0];
    std::vector<float> negative_output_embedding = clip_output[1];

    size_t total_size = output_embedding.size()  + negative_output_embedding.size();
    std::vector<float> combined_embedding(total_size);
    std::copy(output_embedding.begin(), output_embedding.end(), combined_embedding.begin());
    std::copy(negative_output_embedding.begin(), negative_output_embedding.end(), combined_embedding.begin() + output_embedding.size());

    std::vector<half> combined_embedding_fp16(combined_embedding.size());
    std::transform(combined_embedding.begin(), combined_embedding.end(), combined_embedding_fp16.begin(),[](float f) { return __float2half(f);});


    std::vector<half> latents_fp16( 2 * latents.size(),0);

//    auto latents_0 = load_binary_file("/home/lite.ai.toolkit/tensor_data.bin");
//    latents.insert(latents_0.end(), latents_0.begin(), latents_0.end());
//    std::transform(latents_0.begin(), latents_0.end(), latents_fp16.begin(),[](float f) { return __float2half(f);});

    std::transform(latent_input_final.begin(), latent_input_final.end(), latents_fp16.begin(),[](float f) { return __float2half(f);});



    cudaMalloc(&buffers[2], combined_embedding_fp16.size() * sizeof(half));
    trt_context->setTensorAddress(input_names[2],buffers[2]);
    cudaMemcpyAsync(buffers[2], combined_embedding_fp16.data(), combined_embedding_fp16.size() * sizeof(half ),
                    cudaMemcpyHostToDevice, stream);
    nvinfer1::Dims hiddenStateDims;
    hiddenStateDims.nbDims = 3;
    hiddenStateDims.d[0] = input_node_dims_encoder_hidden_states[0];
    hiddenStateDims.d[1] = input_node_dims_encoder_hidden_states[1];
    hiddenStateDims.d[2] = input_node_dims_encoder_hidden_states[2];
    trt_context->setInputShape("encoder_hidden_states", hiddenStateDims);

    cudaMalloc(&buffers[0], latents_fp16.size() * sizeof(half));
    trt_context->setTensorAddress(input_names[0],buffers[0]);
    cudaMemcpyAsync(buffers[0], latents_fp16.data(), latents_fp16.size() * sizeof(half ),
                    cudaMemcpyHostToDevice, stream);
    nvinfer1::Dims sampleDims;
    sampleDims.nbDims = 4;
    sampleDims.d[0] = input_node_dims_sample[0];
    sampleDims.d[1] = input_node_dims_sample[1];
    sampleDims.d[2] = input_node_dims_sample[2];
    sampleDims.d[3] = input_node_dims_sample[3];
    trt_context->setInputShape("sample", sampleDims);

    cudaMalloc(&buffers[3], unet_outputsize * sizeof(half));
    trt_context->setTensorAddress(output_names,buffers[3]);
    std::vector<half> output_trt(unet_outputsize);

    std::vector<float> final_latent_output(final_latent_outputsize,0);
    std::vector<float> pred_sample;

    for (auto t : new_time_steps)
    {
        float t_f = t;
        std::vector<half> t_fp16(1,__float2half(t_f));
        cudaMalloc(&buffers[1], t_fp16.size() * sizeof(half));
        trt_context->setTensorAddress(input_names[1],buffers[1]);
        cudaMemcpyAsync(buffers[1], t_fp16.data(), t_fp16.size() * sizeof(half ),
                        cudaMemcpyHostToDevice, stream);

        nvinfer1::Dims timeStepDims;
        timeStepDims.nbDims = 1;
        timeStepDims.d[0] = 1;
        trt_context->setInputShape("timestep", timeStepDims);


        bool status = trt_context->enqueueV3(stream);

        if (!status){
            std::cerr << "Failed to infer by TensorRT." << std::endl;
            return;
        }

        cudaMemcpyAsync(output_trt.data(), buffers[3], unet_outputsize * sizeof(half),
                        cudaMemcpyDeviceToHost, stream);

        std::vector<float> output_trt_fp32 = convertToFloat(output_trt);


        std::vector<half> noise_pred_uncond(noise_pred_outputsize);
        std::vector<half > noise_pred_text(noise_pred_outputsize);

        std::copy(output_trt.begin(), output_trt.begin() +noise_pred_outputsize, noise_pred_text.begin());
        std::copy(output_trt.begin() + noise_pred_outputsize, output_trt.end(), noise_pred_uncond.begin());


        std::vector<float> noise_pred_uncond_float = convertToFloat(noise_pred_uncond);
        std::vector<float> noise_pred_text_float = convertToFloat(noise_pred_text);


        std::vector<float> noise_pred(noise_pred_outputsize,0);


        float cfgScale = cfg_scale;
        std::transform(noise_pred_uncond_float.begin(),
                       noise_pred_uncond_float.end(),
                       noise_pred_text_float.begin(),
                       noise_pred.begin(),
                       [cfgScale](float uncond, float text) {
                           return uncond + cfgScale * (text - uncond);
                       });

        std::vector<float> latents_fp32 = convertToFloat(latents_fp16);
        scheduler.step(noise_pred,noise_pred_dims, latents_fp32, noise_pred_dims,
                       pred_sample, t);

        std::vector<half> pred_sample_fp16(pred_sample.size(),0);
        std::transform(pred_sample.begin(), pred_sample.end(),
                       pred_sample_fp16.begin(),[](float f) { return __float2half(f);});

        unet_output.assign(pred_sample.begin(),pred_sample.end());

        latents_fp16.clear();
        latents_fp16.assign(pred_sample_fp16.begin(),pred_sample_fp16.end());
        latents_fp16.insert(latents_fp16.end(), latents_fp16.begin(), latents_fp16.end());

        auto temp = convertToFloat(latents_fp16);

        size_t data_size_bytes = latents_fp16.size() * sizeof(half);
        // cudaMemsetAsync 将指定字节数的内存设置为某个值（这里是 0）
        cudaMemsetAsync(buffers[0], 0, data_size_bytes, stream);

        // 3. 异步将 host 数据 (latents_fp16) 拷贝到 device 缓冲区 (buffers[0])
        // 这个拷贝操作会在 stream 上排队，在之前的 memset 操作完成后执行
        cudaMemcpyAsync(buffers[0],                 // 目标 (Device)
                                   latents_fp16.data(),        // 源 (Host)
                                   data_size_bytes,            // 大小 (bytes)
                                   cudaMemcpyHostToDevice,     // 拷贝方向
                                   stream);                   // CUDA 流

        // 4. 同步 CUDA 流
        // 这会阻塞 CPU host 线程，直到 stream 上所有先前提交的操作
        // (包括 cudaMemsetAsync 和 cudaMemcpyAsync) 都已完成。
        cudaStreamSynchronize(stream);

        // update buffers value
//        cudaMemcpyAsync(buffers[0], latents_fp16.data(), latents_fp16.size() * sizeof(half ),
//                        cudaMemcpyHostToDevice, stream);
    }


    std::vector<float> output_trt_float(final_latent_outputsize);
    output_trt_float.assign(final_latent_output.begin(),final_latent_output.end());


}
