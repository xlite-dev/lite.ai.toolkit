#include "cuda_runtime.h"
extern "C" __global__ void nms_kernel(float* bboxes, int number_of_boxes, float threshold_iou, int* result);

extern "C" __device__ float calculate_iou(float* a, float* b);