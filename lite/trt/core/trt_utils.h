//
// Created by ai-test1 on 24-7-11.
//

#ifndef LITE_AI_TOOLKIT_TRT_UTILS_H
#define LITE_AI_TOOLKIT_TRT_UTILS_H
#include "trt_config.h"

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


        }
    }
}


#endif //LITE_AI_TOOLKIT_TRT_UTILS_H
