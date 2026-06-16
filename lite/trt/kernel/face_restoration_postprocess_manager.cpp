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

FaceRestorePostprocessGPU::~FaceRestorePostprocessGPU() {
    if (d_out_) cudaFree(d_out_);
}

const float* FaceRestorePostprocessGPU::run(float* trt_outputs, int channel, int height, int width,
                                            cudaStream_t stream) {
    const int vec_num = channel * height * width;
    const size_t bytes = static_cast<size_t>(vec_num) * sizeof(float);
    if (bytes > cap_) {
        if (d_out_) cudaFree(d_out_);
        cudaMalloc(&d_out_, bytes);
        cap_ = bytes;
    }

    const int block_size = 256;
    const int grid_size = (vec_num + block_size - 1) / block_size;
    face_restoration_postprocess<<<grid_size, block_size, 0, stream>>>(
            trt_outputs, d_out_, channel, height, width);
    return d_out_;   // HWC BGR float[0,255], device-resident; caller consumes on the same stream
}