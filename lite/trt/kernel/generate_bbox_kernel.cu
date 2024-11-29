#include "generate_bbox_kernel.cuh"

__global__ void yolov8_postprocess_kernel(
        float* trt_outputs,
        int number_of_boxes,
        float conf_threshold,
        float ratio_height,
        float ratio_width,
        lite::types::BoundingBoxType<float, float>* output_boxes,  // 直接使用目标类型
        int* output_count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= number_of_boxes) return;

    const float score = trt_outputs[4 * number_of_boxes + index];
    if (score > conf_threshold) {

        float x1 = (trt_outputs[index] - 0.5 * trt_outputs[2 * number_of_boxes + index]) * ratio_width;
        float y1 = (trt_outputs[number_of_boxes + index] - 0.5 * trt_outputs[3 * number_of_boxes + index]) * ratio_height;
        float x2 = (trt_outputs[index] + 0.5 * trt_outputs[2 * number_of_boxes + index]) * ratio_width;
        float y2 = (trt_outputs[number_of_boxes + index] + 0.5 * trt_outputs[3 * number_of_boxes + index]) * ratio_height;

        // 使用原子操作获取输出索引
        int output_index = atomicAdd(output_count, 1);
        // 直接设置BoundingBoxType
        output_boxes[output_index].x1 = x1;
        output_boxes[output_index].y1 = y1;
        output_boxes[output_index].x2 = x2;
        output_boxes[output_index].y2 = y2;
        output_boxes[output_index].score = score;
        output_boxes[output_index].flag = true;
    }
}
