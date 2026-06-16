#include "yoloface_preprocess.cuh"

// One thread per pixel of the letterboxed BGR uint8 image. Writes planar BGR float (CHW),
// normalized v*(1/128) - 127.5/128. Channel order is preserved (plane0=B, plane1=G,
// plane2=R), exactly matching the CPU path (cv::split -> per-channel convertTo -> CHW).
__global__ void yoloface_preprocess_kernel(const unsigned char* img, float* out, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int i = (y * W + x) * 3;
    float b = static_cast<float>(img[i + 0]);
    float g = static_cast<float>(img[i + 1]);
    float r = static_cast<float>(img[i + 2]);

    const float scale = 1.f / 128.f;
    const float shift = -127.5f / 128.f;
    int plane = H * W;
    int off = y * W + x;
    out[0 * plane + off] = b * scale + shift;
    out[1 * plane + off] = g * scale + shift;
    out[2 * plane + off] = r * scale + shift;
}
