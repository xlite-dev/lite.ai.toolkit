#include "paste_back.cuh"
#include <cuda_runtime.h>

__global__ void paste_back_kernel(const float* inverse_vision_frame,
                                  const float* temp_frame,
                                  const float* inverse_mask,
                                  float* output,
                                  int width,
                                  int height,
                                  int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        float inverse_weight = 1.0f - inverse_mask[y * width + x];

        for (int c = 0; c < channels; ++c) {
            output[idx + c] = inverse_mask[y * width + x] * inverse_vision_frame[idx + c] +
                              inverse_weight * temp_frame[idx + c];
        }
    }
}
