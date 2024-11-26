#include <cuda_runtime.h>

struct bbox {
    float x1, y1, x2, y2, score;
};

// IoU计算的device函数
extern "C" __device__ float calculate_iou(float* a, float* b) {
    float left = max(a[0], b[0]);
    float right = min(a[2], b[2]);
    float top = max(a[1], b[1]);
    float bottom = min(a[3], b[3]);

    float width = max(right - left, 0.f);
    float height = max(bottom - top, 0.f);

    float interArea = width * height;
    float boxAArea = (a[2] - a[0]) * (a[3] - a[1]);
    float boxBArea = (b[2] - b[0]) * (b[3] - b[1]);

    return interArea / (boxAArea + boxBArea - interArea);
}

// NMS核函数
extern "C" __global__ void nms_kernel(float* bboxes, int number_of_boxes, float threshold_iou, int* result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查
    if (index >= number_of_boxes) return;

    // 初始假设保留当前框
    result[index] = 1;

    for (int i = 0; i < number_of_boxes; i++) {
        // 跳过自身
        if (i == index) continue;

        // 当前框和比较框的指针
        float* current_box = bboxes + index * 5;
        float* compare_box = bboxes + i * 5;

        float iou = calculate_iou(current_box, compare_box);

        // 获取分数
        float current_score = current_box[4];
        float compare_score = compare_box[4];

        // 如果IoU大于阈值且比较框分数更高，则抑制当前框
        if (iou > threshold_iou && compare_score > current_score) {
            result[index] = 0;
            break;
        }
    }

}
