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

#endif // PASTE_BACK_CUH
