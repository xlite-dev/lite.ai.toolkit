#include "face_restoration_postprocess.cuh"

// 第一步处理函数
__device__ float process_range_single(float x) {
    x = fmax(-1.0f, fmin(1.0f, x));
    return (x + 1.f) / 2.f;
}

// CHW到HWC的索引转换
__device__ int get_hwc_index(int c, int h, int w, int channel, int width) {
    return h * (width * channel) + w * channel + c;
}

// float转uint8的处理
__device__ unsigned char float_to_uint8_simple(float x) {
    return (unsigned char)rintf(fminf(255.f, fmaxf(0.f, x * 255.f)));
}

// 主kernel函数
__global__ void face_restoration_postprocess(
        float* input_buffer,        // 输入数据（TRT输出，CHW格式）
        unsigned char* output_final,  // 最终输出（HWC格式，uint8）
        int channel,
        int height,
        int width
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_size = channel * height * width;
    if (idx >= total_size) return;

    // 第一步：范围处理
    float processed = process_range_single(input_buffer[idx]);

    // 第二步：计算CHW中的位置
    int c = idx / (height * width);
    int h = (idx % (height * width)) / width;
    int w = idx % width;

    // 第三步：计算HWC位置并转换
    int hwc_idx = get_hwc_index(c, h, w, channel, width);

    // 第四步：转换为uint8并写入输出
    output_final[hwc_idx] = float_to_uint8_simple(processed);
}
