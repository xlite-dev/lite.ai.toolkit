#include "cuda_runtime.h"

extern "C" __global__ void bgr2rgb_kernel(const uchar3* bgr, uchar3* rgb, int width, int height);

