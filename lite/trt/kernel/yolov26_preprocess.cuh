//
// Created by zhuohaoyang on 2026/8/28.
//

#ifndef LITE_AI_TOOLKIT_TRT_YOLOV26_PREPROCESS_CUH
#define LITE_AI_TOOLKIT_TRT_YOLOV26_PREPROCESS_CUH

#include <cuda_runtime.h>

#include <cstddef>

namespace trtcv
{
  namespace kernel
  {
    cudaError_t launch_yolov26_preprocess(const unsigned char *src_bgr,
                                          std::size_t src_pitch,
                                          int src_width, int src_height,
                                          float *dst_chw,
                                          int dst_width, int dst_height,
                                          int resized_width, int resized_height,
                                          int left, int top,
                                          cudaStream_t stream);
  }
}

#endif // LITE_AI_TOOLKIT_TRT_YOLOV26_PREPROCESS_CUH
