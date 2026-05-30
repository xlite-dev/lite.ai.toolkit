#ifndef PASTE_BACK_CUH
#define PASTE_BACK_CUH

#include <cuda_runtime.h>

extern "C" __global__ void paste_back_kernel(const float* inverse_vision_frame,
                                             const float* temp_frame,
                                             const float* inverse_mask,
                                             float* output,
                                             int width,
                                             int height,
                                             int channels);

// Fused inverse-mapping paste_back: for each full-frame pixel, map to crop space with M,
// bilinearly sample and blend. temp/out are full-frame BGR uint8; crop is BGR float (0..255);
// mask is float (0..1); M is 6 floats (the original->crop 2x3 affine, row-major).
__global__ void paste_back_fused_kernel(const unsigned char* temp,
                                        const float* crop,
                                        const float* mask,
                                        const float* M,
                                        unsigned char* out,
                                        int W, int H, int Cw, int Ch);

#endif // PASTE_BACK_CUH
