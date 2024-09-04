//
// Created by wangzijian on 8/27/24.
//

#ifndef LITE_AI_TOOLKIT_VAE_H
#define LITE_AI_TOOLKIT_VAE_H
#include "lite/ort/core/ort_core.h"

namespace ortsd
{
    class  Vae
    {
    public:
        Vae(const std::string &_onnx_path, unsigned int _num_threads = 1);

        ~Vae();

    private:
        Ort::Env ort_env;
        Ort::Session *ort_session = nullptr;
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        unsigned int num_inputs = 1;
        const LITEORT_CHAR *onnx_path = nullptr;
        const char *log_id = nullptr;
        bool context_is_update = false;
        const unsigned int num_threads; // initialize at runtime.

        std::vector<const char *> input_node_names = {
                "latent"
        };

        std::vector<const char *> output_node_names = {
                "images"
        };

    public:

        void inference(const std::vector<float> &unet_input,const std::string save_path);

    };
}

#endif //LITE_AI_TOOLKIT_VAE_H
