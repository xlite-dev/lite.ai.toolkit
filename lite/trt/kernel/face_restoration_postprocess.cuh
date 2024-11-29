#include "cuda_runtime.h"
extern "C"  __global__ void face_restoration_postprocess(
        float* input_buffer,        // 输入数据（TRT输出，CHW格式）
        unsigned char* output_final,  // 最终输出（HWC格式，uint8）
        int channel,
        int height,
        int width
);