//
// Created by root on 8/28/24.
//

#include "trt_unet.h"
#include "lite/ort/sd/ddimscheduler.h"
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

void trt_generate_latents(std::vector<float>& latents, int batch_size, int unet_channels, int latent_height, int latent_width, float init_noise_sigma) {
    size_t total_size = batch_size * unet_channels * latent_height * latent_width;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < total_size; ++i) {
        latents[i] = dist(gen) * init_noise_sigma;
    }
}

void trt_save_to_bin(const std::vector<float>& data, const std::string& filename) {
    std::ofstream outfile(filename, std::ios::out | std::ios::binary);
    if (outfile.is_open()) {
        outfile.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
        outfile.close();
    } else {
        std::cerr << "Failed to open file: " << filename << std::endl;
    }
}

std::vector<float> TRTUNet::convertToFloat(const std::vector<half> &half_vec) {
    std::vector<float> float_vec(half_vec.size());
    std::transform(half_vec.begin(), half_vec.end(), float_vec.begin(),
                   [](half h) { return __half2float(h); }); // 使用 __half2float 将 half 转换为 float
    return float_vec;
}

std::vector<half> TRTUNet::convertToHalf(const std::vector<float> &float_vec) {
    std::vector<half> half_vec(float_vec.size());
    std::transform(float_vec.begin(), float_vec.end(), half_vec.begin(),
                   [](float f) { return __float2half(f); }); // 使用 __float2half 将 float 转换为 half
    return half_vec;
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


void TRTUNet::inference() {

    auto start = std::chrono::high_resolution_clock::now();

    // 初始化time step
    auto scheduler = Scheduler::DDIMScheduler("/home/lite.ai.toolkit/lite/ort/sd/scheduler_config.json");
    scheduler.set_timesteps(30);
    std::vector<int> timesteps;
    scheduler.get_timesteps(timesteps);
    auto init_noise_sigma = scheduler.get_init_noise_sigma();
    std::vector<float> latents(1 * 4 * 64 * 64);
    trt_generate_latents(latents, 1, 4, 64, 64, init_noise_sigma);
//    std::string filename = "/home/lite.ai.toolkit/ort_init_latent_data.bin";
//    std::vector<float> latents = trt_load_from_bin(filename);
    latents.insert(latents.end(), latents.begin(), latents.end());


    std::string clip_engine_path = "/home/lite.ai.toolkit/examples/hub/trt/clip_text_model_fp16.engine";
    TRTClip trtClip(clip_engine_path);
    std::vector<std::string> input_text = {"i am not good at cpp"};
    std::vector<std::string> negative_input_text = {""};
    std::vector<std::vector<float>> output_embedding;
    std::vector<std::vector<float>> negative_output_embedding;
    trtClip.inference(input_text, output_embedding);
    trtClip.inference(negative_input_text, negative_output_embedding);
    size_t total_size = output_embedding.size() * output_embedding[0].size() +
                        negative_output_embedding.size() * negative_output_embedding[0].size();
    std::vector<float> combined_embedding(total_size);
    std::copy(output_embedding[0].begin(), output_embedding[0].end(), combined_embedding.begin());
    std::copy(negative_output_embedding[0].begin(), negative_output_embedding[0].end(), combined_embedding.begin() + output_embedding[0].size());


    // 将embedding改为fp16的输入
    std::vector<half> combined_embedding_fp16(combined_embedding.size(),0);
    for (int i = 0; i < combined_embedding_fp16.size(); i++)
    {
        combined_embedding_fp16[i] = __float2half(combined_embedding[i]);
    }

    // 将初始的latent改为fp16的输入
    std::vector<half> latents_fp16(latents.size(),0);
    for (int i = 0; i < latents_fp16.size(); i++)
    {
        latents_fp16[i] = __float2half(latents[i]);
    }

    // 现在的buffer是有三个输入的 输出是一个在外面分配两个 然后在里面分配一个
    // 先分配三个输入的buffer
    cudaMalloc(&buffers[2], combined_embedding_fp16.size() * sizeof(half));
    trt_context->setTensorAddress(input_names[2],buffers[2]);
    cudaMemcpyAsync(buffers[2], combined_embedding_fp16.data(), combined_embedding_fp16.size() * sizeof(half ),
                    cudaMemcpyHostToDevice, stream);
    nvinfer1::Dims hiddenStateDims;
    std::vector<int> hidden_state_dims = {2, 77, 768};  // 根据实际情况调整
    hiddenStateDims.nbDims = 3;
    hiddenStateDims.d[0] = hidden_state_dims[0];
    hiddenStateDims.d[1] = hidden_state_dims[1];
    hiddenStateDims.d[2] = hidden_state_dims[2];
    trt_context->setInputShape("encoder_hidden_states", hiddenStateDims);



    cudaMalloc(&buffers[0], latents_fp16.size() * sizeof(half));
    trt_context->setTensorAddress(input_names[0],buffers[0]);
    cudaMemcpyAsync(buffers[0], latents_fp16.data(), latents_fp16.size() * sizeof(half ),
                    cudaMemcpyHostToDevice, stream);
    std::vector<int> sample_dims = {2, 4, 64, 64};  // 根据实际情况调整
    nvinfer1::Dims sampleDims;
    sampleDims.nbDims = 4;
    sampleDims.d[0] = sample_dims[0];
    sampleDims.d[1] = sample_dims[1];
    sampleDims.d[2] = sample_dims[2];
    sampleDims.d[3] = sample_dims[3];
    trt_context->setInputShape("sample", sampleDims);


    // 设置输出的buffer
    cudaMalloc(&buffers[3], 2 * 4 * 64 * 64 * sizeof(half));
    trt_context->setTensorAddress(output_names,buffers[3]);
    std::vector<half> output_trt(2 * 4 * 64 * 64);

    std::vector<float> final_latent_output(1 * 4 * 64 * 64,0);
    std::vector<float> pred_sample;

    for (auto t : timesteps)
    {
        // 将时间步改为fp16的输入
        float t_f = t;
        std::vector<half> t_fp16(1,__float2half(t_f));
        cudaMalloc(&buffers[1], t_fp16.size() * sizeof(half));
        trt_context->setTensorAddress(input_names[1],buffers[1]);
        cudaMemcpyAsync(buffers[1], t_fp16.data(), t_fp16.size() * sizeof(half ),
                        cudaMemcpyHostToDevice, stream);
        // 为time_step设置维度（假设是标量）
        nvinfer1::Dims timeStepDims;
        timeStepDims.nbDims = 1;
        timeStepDims.d[0] = 1;
        trt_context->setInputShape("timestep", timeStepDims);


        bool status = trt_context->enqueueV3(stream);

        if (!status){
            std::cerr << "Failed to infer by TensorRT." << std::endl;
            return;
        }

        // 将推理处理的结果更新为下一次输入的sample

        // 先将其取出来 注意是fp16的
        cudaMemcpyAsync(output_trt.data(), buffers[3], 2 * 4 * 64 * 64 * sizeof(half),
                        cudaMemcpyDeviceToHost, stream);

        std::vector<half> noise_pred_uncond(1 * 4 * 64 * 64);
        std::vector<half > noise_pred_text(1  * 4 * 64 * 64);

        std::copy(output_trt.begin(), output_trt.begin() + 1 * 4 * 64 * 64, noise_pred_text.begin());
        std::copy(output_trt.begin() + 1 * 4 * 64 * 64, output_trt.end(), noise_pred_uncond.begin());

        // 将 half 类型数据转换为 float 类型数据
        std::vector<float> noise_pred_uncond_float = convertToFloat(noise_pred_uncond);
        std::vector<float> noise_pred_text_float = convertToFloat(noise_pred_text);


        std::vector<float> noise_pred(1 * 4 * 64 * 64,0);
        for (size_t i = 0; i < noise_pred.size(); ++i)
        {
            noise_pred[i] = noise_pred_uncond_float[i] + 7.5f * (noise_pred_text_float[i] - noise_pred_uncond_float[i]);
        }


        std::vector<float> latents_fp32 = convertToFloat(latents_fp16);


        scheduler.step(noise_pred,{1, 4, 64, 64}, latents_fp32, {1, 4, 64, 64},
                       pred_sample, t);


        std::vector<half> pred_sample_fp16(pred_sample.size(),0);
        for (int i = 0; i < pred_sample_fp16.size(); i++)
        {
            pred_sample_fp16[i] = __float2half(pred_sample[i]);
        }

        trt_save_to_bin(pred_sample,"/home/lite.ai.toolkit/trt_final_latent_data.bin");


//        std::vector<half> temp = convertToHalf(pred_sample);
        // 将其拷到latents中

        latents_fp16.clear();
        latents_fp16.assign(pred_sample_fp16.begin(),pred_sample_fp16.end());
        latents_fp16.insert(latents_fp16.end(), latents_fp16.begin(), latents_fp16.end());

        // 需要更新buffer里面的值
        cudaMemcpyAsync(buffers[0], latents_fp16.data(), latents_fp16.size() * sizeof(half ),
                        cudaMemcpyHostToDevice, stream);


    }

    // 现将half2float
    std::vector<float> output_trt_float(1 * 4 * 64 * 64);


    output_trt_float.assign(final_latent_output.begin(),final_latent_output.end());
    // 将最后一次的结果保存下来 放到vae里面去推理

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "没有加载engine的耗时是: " << duration << " 毫秒" << std::endl;



}