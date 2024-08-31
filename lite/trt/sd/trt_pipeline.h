//
// Created by wangzijian on 8/31/24.
//

#ifndef LITE_AI_TOOLKIT_TRT_PIPELINE_H
#define LITE_AI_TOOLKIT_TRT_PIPELINE_H
#include "lite/trt/core/trt_core.h"
#include "lite/trt/sd/trt_vae.h"
#include "lite/trt/sd/trt_clip.h"
#include "lite/trt/sd/trt_unet.h"


namespace trtsd{

    class TRTPipeline {
    public:
        TRTPipeline(const std::string &_clip_engine_path,
                 const std::string &_unet_engine_path,
                 const std::string &_vae_engine_path);
        ~TRTPipeline() = default;

    private:
        std::unique_ptr<TRTUNet> unet;
        std::unique_ptr<TRTClip> clip;
        std::unique_ptr<TRTVae> vae;

    public:
        void inference(std::string prompt,std::string negative_prompt,std::string image_save_path);
    };

}


#endif //LITE_AI_TOOLKIT_TRT_PIPELINE_H
