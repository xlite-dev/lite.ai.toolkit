//
// Created by root on 12/2/24.
//

#include "face_swap_postproces_manager.h"
void launch_face_swap_postprocess( float *input,// 这里是trt的输出
                                   int channel,
                                   int height,
                                   int width,
                                   float *output){
    int block_size = 256;
    auto size = channel * width * height;
    int grid_size =  (size + block_size -1 ) / block_size;
    float* d_output;
    cudaMalloc(&d_output,size * sizeof (float ));
    face_swap_postprocess<<<grid_size,block_size>>>(
            input,channel,height,width,d_output
            );
    // 将生成的拷贝到host端
    cudaMemcpy(output,d_output,size * sizeof(float ),cudaMemcpyDeviceToHost);
    cudaFree(d_output);
}