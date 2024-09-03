//
// Created by wangzijian on 8/14/24.
//

#ifndef LITE_AI_TOOLKIT_UNET_H
#define LITE_AI_TOOLKIT_UNET_H
#include "iostream"
#include "lite/ort/core/ort_core.h"
#include "ddimscheduler.h"
#include "clip.h"
#include <random>
#include <cstdint>
#include <iomanip>
#include "tokenizer.h"


namespace ortsd
{
    class  UNet
    {
    public:
        UNet(const std::string &_onnx_path, unsigned int _num_threads = 1);

        ~UNet();

    private:
        Ort::Env ort_env;
        Ort::Session *ort_session = nullptr;
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);


        unsigned int num_inputs = 3;
        const LITEORT_CHAR *onnx_path = nullptr;
        const char *log_id = nullptr;
        bool context_is_update = false;
        const unsigned int num_threads; // initialize at runtime.
        const int encoder_input_size = 77 * 768;

        std::vector<const char *> input_node_names = {
                "sample",
                "timestep",
                "encoder_hidden_states"
        };

        std::vector<const char *> output_node_names = {
                "latent"
        };

    public:

        void inference(std::vector<std::vector<float>> clip_output,std::vector<float> &latent,std::string scheduler_config_path);


    };
}


#endif //LITE_AI_TOOLKIT_UNET_H
