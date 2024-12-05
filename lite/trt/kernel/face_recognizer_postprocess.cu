#include "face_recognizer_postprocess.cuh"


__global__ void computeNormKernel(
        float* input,
        int size,
        float* norm
){
    // 将线程块大小设置为256
    __shared__ float sharedMem[256];
    // 每个线程单独的本地的norm的
    float localSum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 每个线程计算自己的平方和
    if (idx < size) {
        localSum = input[idx] * input[idx];
        // CUDA kernel print (works within kernel)
//        printf("Thread %d: localSum = %f\n", idx, input[idx]);
    }
    // 贡献自己的平方和给共享的变量

    sharedMem[threadIdx.x] = localSum;

    // 并行归约
    // 计算共享内存中的和 也就是norm的值
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (threadIdx.x < stride) {
            sharedMem[threadIdx.x] += sharedMem[threadIdx.x + stride];
        }
    }

    // 利用0号线程写回结果
    if (threadIdx.x == 0) {
        *norm = sqrtf(sharedMem[0]);
//        printf("Thread %d: norm = %f\n", idx,*norm);
    }
}

__global__ void normalizeVectorKernel(float* input, int size, float norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
//        printf("Thread %d: norm = %f\n", idx,norm);
        input[idx] /= norm;
    }
}
