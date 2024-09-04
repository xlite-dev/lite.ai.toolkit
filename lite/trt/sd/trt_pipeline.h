//
// Created by wangzijian on 8/31/24.
//

#ifndef LITE_AI_TOOLKIT_TRT_PIPELINE_H
#define LITE_AI_TOOLKIT_TRT_PIPELINE_H
#include "lite/trt/core/trt_core.h"
#include "lite/trt/sd/trt_vae.h"
#include "lite/trt/sd/trt_clip.h"
#include "lite/trt/sd/trt_unet.h"

namespace trtsd {

    class TRTPipeline {
    public:
        TRTPipeline(const std::string &_clip_engine_path,
                    const std::string &_unet_engine_path,
                    const std::string &_vae_engine_path,
                    bool is_low_vram = true);
        ~TRTPipeline() = default;

    private:
        std::string clip_engine_path;
        std::string unet_engine_path;
        std::string vae_engine_path;

        std::unique_ptr<TRTUNet> unet = nullptr;
        std::unique_ptr<TRTClip> clip = nullptr;
        std::unique_ptr<TRTVae> vae = nullptr;

    public:
        void inference(std::string prompt, std::string negative_prompt, std::string image_save_path, std::string scheduler_config_path,bool is_low_vram = true);
    };

}


#endif //LITE_AI_TOOLKIT_TRT_PIPELINE_H