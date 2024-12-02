//
// Created by root on 11/30/24.
//

#include "face_recognizer_postprocess_manager.h"
#include "iostream"
#include "vector"
void launch_face_recognizer_postprocess( float* input_buffer,int size,
                                         float* output_embedding){

    // 在这里调用两个Kernel完成操作
    int block_size = 256;
    int grid_size = (size + block_size -1 ) / block_size;

    float h_norm = 0.0f;
    float* d_output_embedding;
    float* d_norm;

    // 分配GPU上的空间
    cudaMalloc(&d_output_embedding,size * sizeof(float ));
    cudaMalloc(&d_norm,1 * sizeof(float ));


    cudaMemcpy(d_output_embedding, input_buffer,
               sizeof (float ) * size,
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_norm, &h_norm,
               sizeof (float ) * 1,
               cudaMemcpyHostToDevice);
    // 这里是因为如果分开的话 那么数组归约会失败 这个需要想个办法解决掉
    computeNormKernel<<<1,512>>>(d_output_embedding,size,d_norm);
    cudaDeviceSynchronize(); // 确保norm计算完成
    float h_norm1 = 0.0f;

    cudaMemcpy(&h_norm1,d_norm,1 * sizeof (float ),cudaMemcpyDeviceToHost);
    normalizeVectorKernel<<<grid_size,block_size>>>(
            d_output_embedding,
            size,
            h_norm1
            );
    // 执行完上面两步拷贝出来
    cudaMemcpy(input_buffer,d_output_embedding,size * sizeof(float ),
               cudaMemcpyDeviceToHost);
    cudaFree(d_output_embedding);
}