//
// Created by root on 11/29/24.
//

#include "face_restoration_postprocess_manager.h"
void launch_face_restoration_postprocess(
        float* trt_outputs,
        float* output_final,        // HWC, BGR, float [0,255]
        int channel,
        int height,
        int width
){
    int block_size  = 256;
    int vec_num = channel * height * width;
    int grid_size = ( vec_num + block_size - 1) / block_size;

    float* d_output_final;
    cudaMalloc(&d_output_final, vec_num * sizeof(float));

    face_restoration_postprocess<<<grid_size,block_size>>>(
            trt_outputs,
            d_output_final,
            channel,
            height,
            width
            );
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    cudaMemcpy(output_final, d_output_final, vec_num * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(d_output_final);
}