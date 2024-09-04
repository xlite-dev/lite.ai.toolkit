//
// Created by root on 8/31/24.
//

#include "pipeline.h"
using ortsd::Pipeline;

Pipeline::Pipeline(const std::string &_clip_onnx_path, const std::string &_unet_onnx_path,
                   const std::string &_vae_onnx_path, unsigned int _num_threads){
    // 在这里初始化三个类
    // 三个智能指针
    clip = std::make_unique<Clip>(_clip_onnx_path, _num_threads);
    unet = std::make_unique<UNet>(_unet_onnx_path, _num_threads);
    vae = std::make_unique<Vae>(_vae_onnx_path, _num_threads);
}


void Pipeline::inference(std::string prompt, std::string negative_prompt, std::string image_save_path, std::string scheduler_config_path) {
//    clip->inference()

        std::vector<std::string> total_prompt = {std::move(prompt), std::move(negative_prompt)};

        std::vector<std::vector<float>> clip_output;
        clip->inference(total_prompt,clip_output);
        // 删除clip对象
        clip.reset();


        // 得到clip的输出
        // unet inference
        std::vector<float> unet_output;
        unet->inference(clip_output,unet_output,scheduler_config_path);
        unet.reset();

        // 得到unet的输出
        // vae inference
        vae->inference(unet_output,std::move(image_save_path));
        vae.reset();
}