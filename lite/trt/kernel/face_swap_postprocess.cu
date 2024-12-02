#include "face_swap_postprocess.cuh"

// CHW到HWC的索引转换
__device__ int get_hwc_index_test(int c, int h, int w, int channel, int width) {
    return h * (width * channel) + w * channel + c;
}

__global__ void face_swap_postprocess(float* input,int channel,int height,int width,
                                      float* output){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int size = channel * height * width;
    if (idx > size) return;

    int c = idx / (height * width);
    int h = (idx % (height * width)) / width;
    int w = idx % width;
    // 第三步：计算HWC位置并转换
    int hwc_idx = get_hwc_index_test(c, h, w, channel, width);

    output[hwc_idx] = roundf(input[idx] * 255.f);
}