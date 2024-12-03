#include "bgr2rgb.cuh"

__global__ void bgr2rgb_kernel(const uchar3* bgr, uchar3* rgb, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int i = y * width + x;
        rgb[i].x = bgr[i].z;
        rgb[i].y = bgr[i].y;
        rgb[i].z = bgr[i].x;
    }
}
