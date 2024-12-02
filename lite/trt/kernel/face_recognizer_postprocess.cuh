#include "cuda_runtime.h"
#include "stdio.h"
extern "C" __global__ void computeNormKernel(
        float* input,
        int size,
        float* norm
);

extern "C" __global__ void normalizeVectorKernel(float* input,
                                                 int size, float norm);