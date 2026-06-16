//
// Created by root on 11/29/24.
//

#ifndef LITE_AI_TOOLKIT_FACE_RESTORATION_POSTPROCESS_MANAGER_H
#define LITE_AI_TOOLKIT_FACE_RESTORATION_POSTPROCESS_MANAGER_H
#include <vector>
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>
#include "face_restoration_postprocess.cuh"

void launch_face_restoration_postprocess(
        float* trt_outputs,
        float* output_final,        // HWC, BGR, float [0,255]
        int channel,
        int height,
        int width
        );

// Device-resident variant: runs the postprocess kernel into a reusable device buffer (no per-call
// cudaMalloc, no D2H) and returns the device pointer to the HWC BGR float[0,255] crop, ready to
// feed paste-back directly. Launches on `stream` and does NOT sync.
class FaceRestorePostprocessGPU {
public:
    FaceRestorePostprocessGPU() = default;
    ~FaceRestorePostprocessGPU();
    FaceRestorePostprocessGPU(const FaceRestorePostprocessGPU&) = delete;
    FaceRestorePostprocessGPU& operator=(const FaceRestorePostprocessGPU&) = delete;

    const float* run(float* trt_outputs, int channel, int height, int width,
                     cudaStream_t stream = nullptr);

private:
    float* d_out_ = nullptr;
    size_t cap_ = 0;
};


#endif //LITE_AI_TOOLKIT_FACE_RESTORATION_POSTPROCESS_MANAGER_H
