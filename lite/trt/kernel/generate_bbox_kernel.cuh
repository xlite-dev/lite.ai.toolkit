#include "cuda_runtime.h"
#include "lite/types.h"

extern "C" __global__ void yolov8_postprocess_kernel(
        float* trt_outputs,
        int number_of_boxes,
        float conf_threshold,
        float ratio_height,
        float ratio_width,
        lite::types::BoundingBoxType<float, float>* output_boxes,  // 直接使用目标类型
        int* output_count
);
