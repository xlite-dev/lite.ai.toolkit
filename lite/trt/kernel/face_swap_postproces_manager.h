//
// Created by root on 12/2/24.
//
#include "face_swap_postprocess.cuh"
#ifndef LITE_AI_TOOLKIT_FACE_SWAP_POSTPROCES_MANAGER_H
#define LITE_AI_TOOLKIT_FACE_SWAP_POSTPROCES_MANAGER_H

void launch_face_swap_postprocess(
        float *input,// 这里是trt的输出
        int channel,
        int height,
        int width,
        float *output
        );

#endif //LITE_AI_TOOLKIT_FACE_SWAP_POSTPROCES_MANAGER_H
