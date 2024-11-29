//
// Created by root on 11/29/24.
//

#include "face_restoration_postprocess_manager.h"
void launch_face_restoration_postprocess(
        float* trt_outputs,
        unsigned char* output_final,
        int channel,
        int height,
        int width
){
    // 设计grid和block的尺寸 block直接设置为256的最大值
    int block_size  = 256;
    int vec_num = channel * height * width;
    int grid_size = ( vec_num + block_size - 1) / block_size;
    // GPU上的内存空间
    unsigned char* d_output_final;
    int* d_output_count;

    // 在GPU上分配输出的空间
    cudaMalloc(&d_output_final,vec_num * sizeof(unsigned char ));

    // 启动内核
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

    // 将生成的数据复制出来
    cudaMemcpy(output_final,d_output_final,vec_num * sizeof(unsigned char ),
               cudaMemcpyDeviceToHost);

    // 释放cuda上的内存
    cudaFree(d_output_final);

}