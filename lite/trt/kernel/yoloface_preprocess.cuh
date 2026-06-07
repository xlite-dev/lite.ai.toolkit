#ifndef YOLOFACE_PREPROCESS_CUH
#define YOLOFACE_PREPROCESS_CUH

#include <cuda_runtime.h>

// One thread per pixel of the letterboxed BGR uint8 image. Writes planar BGR float (CHW),
// normalized v*(1/128) - 127.5/128, matching the CPU normalize() in trt_yolofacev8.cpp.
__global__ void yoloface_preprocess_kernel(const unsigned char* img, float* out, int H, int W);

#endif // YOLOFACE_PREPROCESS_CUH
