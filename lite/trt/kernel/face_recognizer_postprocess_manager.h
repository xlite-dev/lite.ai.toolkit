//
// Created by root on 11/30/24.
//

#ifndef LITE_AI_TOOLKIT_FACE_RECOGNIZER_POSTPROCESS_MANAGER_H
#define LITE_AI_TOOLKIT_FACE_RECOGNIZER_POSTPROCESS_MANAGER_H
#include "face_recognizer_postprocess.cuh"
void launch_face_recognizer_postprocess(
        float* input_buffer,
        int size,
        float* output_embedding
);


#endif //LITE_AI_TOOLKIT_FACE_RECOGNIZER_POSTPROCESS_MANAGER_H
