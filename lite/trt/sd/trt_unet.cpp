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


void TRTUNet::inference() {
    // 初始化time step
    auto scheduler = Scheduler::DDIMScheduler("/home/lite.ai.toolkit/lite/ort/sd/scheduler_config.json");
    scheduler.set_timesteps(30);
    std::vector<int> timesteps;
    scheduler.get_timesteps(timesteps);
    auto init_noise_sigma = scheduler.get_init_noise_sigma();
    std::vector<float> latents(1 * 4 * 64 * 64);
    trt_generate_latents(latents, 1, 4, 64, 64, init_noise_sigma);
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
    cudaMalloc(&buffers[0], combined_embedding_fp16.size() * sizeof(half));
    trt_context->setTensorAddress(input_names[0],buffers[0]);
    cudaMemcpyAsync(buffers[0], combined_embedding_fp16.data(), combined_embedding_fp16.size() * sizeof(half ),
                    cudaMemcpyHostToDevice, stream);

    std::vector<int> sample_dims = {2, 4, 64, 64};  // 根据实际情况调整
    nvinfer1::Dims sampleDims;
    sampleDims.nbDims = 4;
    sampleDims.d[0] = sample_dims[0];
    sampleDims.d[1] = sample_dims[1];
    sampleDims.d[2] = sample_dims[2];
    sampleDims.d[3] = sample_dims[3];
    trt_context->setInputShape("sample", sampleDims);


    cudaMalloc(&buffers[1], latents_fp16.size() * sizeof(half));
    trt_context->setTensorAddress(input_names[1],buffers[1]);
    cudaMemcpyAsync(buffers[1], latents_fp16.data(), latents_fp16.size() * sizeof(half ),
                    cudaMemcpyHostToDevice, stream);
    nvinfer1::Dims hiddenStateDims;
    std::vector<int> hidden_state_dims = {2, 77, 768};  // 根据实际情况调整
    hiddenStateDims.nbDims = 3;
    hiddenStateDims.d[0] = hidden_state_dims[0];
    hiddenStateDims.d[1] = hidden_state_dims[1];
    hiddenStateDims.d[2] = hidden_state_dims[2];
    trt_context->setInputShape("encoder_hidden_states", hiddenStateDims);



    // 设置输出的buffer
    cudaMalloc(&buffers[3], 2 * 4 * 64 * 64 * sizeof(half));
    trt_context->setTensorAddress(output_names,buffers[3]);


    for (auto t : timesteps)
    {
        // 将时间步改为fp16的输入
        float t_f = t;
        std::vector<half> t_fp16(1,__float2half(t_f));
        cudaMalloc(&buffers[2], t_fp16.size() * sizeof(half));
        trt_context->setTensorAddress(input_names[2],buffers[2]);
        cudaMemcpyAsync(buffers[2], t_fp16.data(), t_fp16.size() * sizeof(half ),
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
        std::vector<half> output_trt(2 * 4 * 64 * 64);






    }





}