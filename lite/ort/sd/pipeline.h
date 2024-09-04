//
// Created by root on 8/31/24.
//

#ifndef LITE_AI_TOOLKIT_PIPELINE_H
#define LITE_AI_TOOLKIT_PIPELINE_H

#include "lite/ort/core/ort_core.h"
#include "lite/ort/sd/vae.h"
#include "lite/ort/sd/clip.h"
#include "lite/ort/sd/unet.h"

namespace ortsd{

    class Pipeline{

    public:
        Pipeline(const std::string &_clip_onnx_path,
                 const std::string &_unet_onnx_path,
                 const std::string &_vae_onnx_path,
                 unsigned int _num_threads = 1);

        ~Pipeline() = default; // 默认析构函数，因为智能指针会自动清理

    private:
        std::unique_ptr<Clip> clip;
        std::unique_ptr<UNet> unet;
        std::unique_ptr<Vae> vae;


    public:

        void inference(std::string prompt,std::string negative_prompt,std::string image_save_path,std::string scheduler_config_path);


    };

}



#endif //LITE_AI_TOOLKIT_PIPELINE_H
