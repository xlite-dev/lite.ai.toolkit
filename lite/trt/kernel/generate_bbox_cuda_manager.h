//
// Created by wangzijian on 11/26/24.
//

#pragma once
#include <vector>
#include <memory>
#include <stdexcept>
#include "lite/types.h"
#include "generate_bbox_kernel.cuh"

void launch_yolov8_postprocess(
        float* trt_outputs,
        int number_of_boxes,
        float conf_threshold,
        float ratio_height,
        float ratio_width,
        lite::types::BoundingBoxType<float, float>* output_boxes,
        int max_output_boxes
);