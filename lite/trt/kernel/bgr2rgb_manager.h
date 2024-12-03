//
// Created by root on 12/3/24.
//

#ifndef LITE_AI_TOOLKIT_BGR2RGB_MANAGER_H
#define LITE_AI_TOOLKIT_BGR2RGB_MANAGER_H
#include "bgr2rgb.cuh"
#include "opencv2/opencv.hpp"

void launch_bgr2rgb(const cv::Mat& bgr_image, cv::Mat& rgb_image);


#endif //LITE_AI_TOOLKIT_BGR2RGB_MANAGER_H
