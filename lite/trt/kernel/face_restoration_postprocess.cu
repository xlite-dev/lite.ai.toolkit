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
        float* input_buffer,        // 输入数据（TRT输出，CHW格式，RGB）
        float* output_final,        // output: HWC, BGR, float in [0,255]
        int channel,
        int height,
        int width
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_size = channel * height * width;
    if (idx >= total_size) return;

    // clamp + (x+1)/2 -> [0,1]
    float processed = process_range_single(input_buffer[idx]);

    // CHW position
    int c = idx / (height * width);
    int h = (idx % (height * width)) / width;
    int w = idx % width;

    // Write directly as HWC, BGR, float in [0,255]: this folds the old CPU uint8->float
    // conversion AND the cv::cvtColor(RGB2BGR) into the kernel. Model channels are RGB
    // (0,1,2) -> BGR positions (2,1,0).
    int out_c = channel - 1 - c;
    int hwc_idx = get_hwc_index(out_c, h, w, channel, width);
    output_final[hwc_idx] = processed * 255.f;
}
