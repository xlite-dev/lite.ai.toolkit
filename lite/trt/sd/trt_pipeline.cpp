#include "trt_pipeline.h"
using trtsd::TRTPipeline;

TRTPipeline::TRTPipeline(const std::string &_clip_engine_path, const std::string &_unet_engine_path,
                         const std::string &_vae_engine_path):
         clip_engine_path(_clip_engine_path),
          unet_engine_path(_unet_engine_path),
          vae_engine_path(_vae_engine_path) {
    // 初始时不分配资源，将成员变量置为 nullptr
}

void TRTPipeline::inference(std::string prompt, std::string negative_prompt, std::string image_save_path, std::string scheduler_config_path) {

    // 如果 clip 为空则初始化
    if (!clip) {
        clip = std::make_unique<TRTClip>(clip_engine_path);
    }
    std::vector<std::string> total_prompt = {std::move(prompt), std::move(negative_prompt)};
    std::vector<std::vector<float>> clip_output;
    clip->inference(total_prompt, clip_output);
    clip.reset(); // 使用后立即释放

    // 如果 unet 为空则初始化
    if (!unet) {
        unet = std::make_unique<TRTUNet>(unet_engine_path);
    }
    std::vector<float> unet_output;
    unet->inference(clip_output, unet_output, scheduler_config_path);
    unet.reset(); // 使用后立即释放

    // 如果 vae 为空则初始化
    if (!vae) {
        vae = std::make_unique<TRTVae>(vae_engine_path);
    }
    vae->inference(unet_output, std::move(image_save_path));
    vae.reset(); // 使用后立即释放
}