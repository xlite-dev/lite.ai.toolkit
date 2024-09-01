//
// Created by root on 8/31/24.
//

#include "trt_pipeline.h"
using trtsd::TRTPipeline;

TRTPipeline::TRTPipeline(const std::string &_clip_engine_path, const std::string &_unet_engine_path,
                         const std::string &_vae_engine_path) {

        //    clip = std::make_unique<TRTClip>(_clip_engine_path);
        //    unet = std::make_unique<TRTUNet>(_unet_engine_path);
        //    vae = std::make_unique<TRTVae>(_vae_engine_path);

}

void TRTPipeline::inference(std::string prompt, std::string negative_prompt, std::string image_save_path) {

    clip = std::make_unique<TRTClip>("/home/wangzijian/lite.ai.toolkit/examples/hub/trt/clip_model_fp16.engine");
    std::vector<std::string> total_prompt = {std::move(prompt), std::move(negative_prompt)};
    std::vector<std::vector<float>> clip_output;
    clip->inference(total_prompt,clip_output);
    clip.reset();

    unet = std::make_unique<TRTUNet>("/home/wangzijian/lite.ai.toolkit/examples/hub/trt/unet_model_fp16.engine");
    std::vector<float> unet_output;
    unet->inference(clip_output,unet_output);
    unet.reset();

    vae = std::make_unique<TRTVae>("/home/wangzijian/lite.ai.toolkit/examples/hub/trt/vae_model_fp16.engine");
    vae->inference(unet_output,std::move(image_save_path));
    vae.reset();
}
