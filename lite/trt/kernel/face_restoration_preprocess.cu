#include "face_restoration_preprocess.cuh"

// One thread per crop pixel. Reads interleaved BGR uint8, writes planar RGB float (CHW),
// normalized to [-1, 1] (v/127.5 - 1). Channel mapping: R->plane0, G->plane1, B->plane2.
__global__ void face_restoration_preprocess_kernel(const unsigned char* crop, float* out, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int i = (y * W + x) * 3;
    float b = static_cast<float>(crop[i + 0]);
    float g = static_cast<float>(crop[i + 1]);
    float r = static_cast<float>(crop[i + 2]);

    int plane = H * W;
    int off = y * W + x;
    out[0 * plane + off] = r / 127.5f - 1.f;
    out[1 * plane + off] = g / 127.5f - 1.f;
    out[2 * plane + off] = b / 127.5f - 1.f;
}
