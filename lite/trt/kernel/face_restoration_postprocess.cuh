#include "cuda_runtime.h"
extern "C"  __global__ void face_restoration_postprocess(
        float* input_buffer,        // 输入数据（TRT输出，CHW格式，RGB）
        float* output_final,        // output: HWC, BGR, float [0,255]
        int channel,
        int height,
        int width
);