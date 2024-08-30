//
// Created by ai-test1 on 24-7-11.
//

#ifndef LITE_AI_TOOLKIT_TRT_UTILS_H
#define LITE_AI_TOOLKIT_TRT_UTILS_H
#include "trt_config.h"
#include <random>
#include <algorithm>

namespace trtcv
{
    // specific utils for trt .
    namespace utils
    {
        namespace transform
        {
            enum
            {
                CHW = 0, HWC =1
            };
            LITE_EXPORTS void create_tensor(const cv::Mat &mat,std::vector<float> &input_vector,std::vector<int64_t> input_node_dims,unsigned int data_format = CHW);

            LITE_EXPORTS std::vector<float> trt_load_from_bin(const std::string& filename);

            LITE_EXPORTS void trt_save_to_bin(const std::vector<float>& data, const std::string& filename);

            LITE_EXPORTS void trt_generate_latents(std::vector<float>& latents, int batch_size, int unet_channels, int latent_height, int latent_width, float init_noise_sigma);
        }
    }
}


#endif //LITE_AI_TOOLKIT_TRT_UTILS_H
