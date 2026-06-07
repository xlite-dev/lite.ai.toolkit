#ifndef FACE_RESTORATION_PREPROCESS_CUH
#define FACE_RESTORATION_PREPROCESS_CUH

#include <cuda_runtime.h>

// Fused face-restoration preprocess: takes the HxW interleaved BGR uint8 crop and writes a
// CHW (3,H,W) float tensor that is RGB and normalized by v/127.5 - 1 (i.e. (v/255)*2 - 1).
__global__ void face_restoration_preprocess_kernel(const unsigned char* crop, float* out, int H, int W,
                                                   float scale, float bias);

#endif // FACE_RESTORATION_PREPROCESS_CUH
