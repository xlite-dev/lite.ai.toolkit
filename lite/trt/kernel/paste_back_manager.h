#ifndef PASTE_BACK_MANAGER_H
#define PASTE_BACK_MANAGER_H

#include "paste_back.cuh"
#include <opencv2/opencv.hpp>

cv::Mat launch_paste_back(const cv::Mat& temp_vision_frame,
                          const cv::Mat& crop_vision_frame,
                          const cv::Mat& crop_mask,
                          const cv::Mat& affine_matrix);

#endif // PASTE_BACK_MANAGER_H
