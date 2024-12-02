#include "cuda_runtime.h"
extern "C" __global__ void face_swap_postprocess(float* input,int channel,int height,int width,
                                      float* output);