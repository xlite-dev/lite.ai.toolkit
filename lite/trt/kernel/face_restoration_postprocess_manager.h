//
// Created by root on 11/29/24.
//

#ifndef LITE_AI_TOOLKIT_FACE_RESTORATION_POSTPROCESS_MANAGER_H
#define LITE_AI_TOOLKIT_FACE_RESTORATION_POSTPROCESS_MANAGER_H
#include <vector>
#include <memory>
#include <stdexcept>
#include "face_restoration_postprocess.cuh"

void launch_face_restoration_postprocess(
        float* trt_outputs,
        unsigned char* output_final,
        int channel,
        int height,
        int width
        );


#endif //LITE_AI_TOOLKIT_FACE_RESTORATION_POSTPROCESS_MANAGER_H
