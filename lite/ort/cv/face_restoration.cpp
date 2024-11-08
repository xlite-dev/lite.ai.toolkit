//
// Created by wangzijian on 11/7/24.
//

#include "face_restoration.h"

using ortcv::Face_Restoration;


cv::Mat Face_Restoration::paste_back(const cv::Mat& temp_vision_frame,
                   const cv::Mat& crop_vision_frame,
                   const cv::Mat& crop_mask,
                   const cv::Mat& affine_matrix) {
    // 确保所有图像都是float类型
    cv::Mat temp_float, crop_float, mask_float;
    temp_vision_frame.convertTo(temp_float, CV_32F);
    crop_vision_frame.convertTo(crop_float, CV_32F);
    crop_mask.convertTo(mask_float, CV_32F);

    // 获取仿射变换的逆矩阵
    cv::Mat inverse_matrix;
    cv::invertAffineTransform(affine_matrix, inverse_matrix);

    // 获取目标尺寸
    cv::Size temp_size(temp_vision_frame.cols, temp_vision_frame.rows);

    // 对mask进行反向仿射变换
    cv::Mat inverse_mask;
    cv::warpAffine(mask_float, inverse_mask, inverse_matrix, temp_size);
    cv::threshold(inverse_mask, inverse_mask, 1.0, 1.0, cv::THRESH_TRUNC); // clip at 1
    cv::threshold(inverse_mask, inverse_mask, 0.0, 0.0, cv::THRESH_TOZERO); // clip at 0

    // 对crop_vision_frame进行反向仿射变换
    cv::Mat inverse_vision_frame;
    cv::warpAffine(crop_float, inverse_vision_frame, inverse_matrix,
                   temp_size, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    // 创建输出图像
    cv::Mat paste_vision_frame;
    temp_float.copyTo(paste_vision_frame);

    // 对每个通道进行混合
    std::vector<cv::Mat> channels(3);
    std::vector<cv::Mat> inverse_channels(3);
    std::vector<cv::Mat> temp_channels(3);

    cv::split(inverse_vision_frame, inverse_channels);
    cv::split(temp_float, temp_channels);

    // 创建 1 - mask
    cv::Mat inverse_weight;
    cv::subtract(cv::Scalar(1.0), inverse_mask, inverse_weight);

    for (int i = 0; i < 3; ++i) {
        // 确保所有运算都在相同类型（CV_32F）下进行
        cv::Mat weighted_inverse, weighted_temp;
        cv::multiply(inverse_mask, inverse_channels[i], weighted_inverse);
        cv::multiply(inverse_weight, temp_channels[i], weighted_temp);
        cv::add(weighted_inverse, weighted_temp, channels[i]);
    }

    cv::merge(channels, paste_vision_frame);

    // 如果需要，将结果转换回原始类型
    cv::Mat result;
    if(temp_vision_frame.type() != CV_32F) {
        paste_vision_frame.convertTo(result, temp_vision_frame.type());
    } else {
        result = paste_vision_frame;
    }

    return result;
}


cv::Mat Face_Restoration::create_static_box_mask(std::vector<float> crop_size) {
    // Calculate blur parameters
    int blur_amount = static_cast<int>(crop_size[0] * 0.5 * face_mask_blur);
    int blur_area = std::max(blur_amount / 2, 1);

    // Create initial mask filled with ones
    cv::Mat box_mask = cv::Mat::ones(crop_size[1], crop_size[0], CV_32F);

    // Calculate padding areas
    int top_padding = std::max(blur_area, static_cast<int>(crop_size[1] * face_mask_padding[0] / 100.0));
    int bottom_padding = std::max(blur_area, static_cast<int>(crop_size[1] * face_mask_padding[2] / 100.0));
    int right_padding = std::max(blur_area, static_cast<int>(crop_size[0] * face_mask_padding[1] / 100.0));
    int left_padding = std::max(blur_area, static_cast<int>(crop_size[0] * face_mask_padding[3] / 100.0));

    // Set padding regions to zero
    // Top region
    if (top_padding > 0) {
        box_mask(cv::Rect(0, 0, crop_size[0], top_padding)) = 0.0;
    }

    // Bottom region
    if (bottom_padding > 0) {
        box_mask(cv::Rect(0, crop_size[1] - bottom_padding, crop_size[0], bottom_padding)) = 0.0;
    }

    // Left region
    if (left_padding > 0) {
        box_mask(cv::Rect(0, 0, left_padding, crop_size[1])) = 0.0;
    }

    // Right region
    if (right_padding > 0) {
        box_mask(cv::Rect(crop_size[0] - right_padding, 0, right_padding, crop_size[1])) = 0.0;
    }

    // Apply Gaussian blur if needed
    if (blur_amount > 0) {
        cv::GaussianBlur(box_mask, box_mask, cv::Size(0, 0), blur_amount * 0.25);
    }

    return box_mask;
}


std::pair<cv::Mat, cv::Mat>
Face_Restoration::warp_face_by_face_landmark_5(cv::Mat input_mat, std::vector<cv::Point2f> face_landmark_5) {

    // 创建标准模板点
    std::vector<cv::Point2f> normed_template;
    for(auto current_template : face_template)  // face_template应该是类的成员变量
    {
        current_template.x = current_template.x * 512;  // 512
        current_template.y = current_template.y * 512;  // 注意：原代码中y使用了x，这里修正为y
        normed_template.emplace_back(current_template);
    }

    // 估计仿射变换矩阵
    cv::Mat inliers;
    cv::Mat affine_matrix = cv::estimateAffinePartial2D(
            face_landmark_5,      // 源点
            normed_template,      // 目标点
            inliers,             // 内点掩码
            cv::RANSAC,          // 方法
            100                  // ransacReprojThreshold
    );

    // 检查变换矩阵是否有效
    if (affine_matrix.empty()) {
        throw std::runtime_error("Failed to estimate affine transformation");
    }

    // 进行仿射变换
    cv::Mat crop_img;
    cv::warpAffine(
            input_mat,           // 输入图像
            crop_img,           // 输出图像
            affine_matrix,      // 变换矩阵
            cv::Size(512, 512), // 输出大小
            cv::INTER_AREA,     // 插值方法
            cv::BORDER_REPLICATE // 边界处理方式
    );

    // 返回结果对
    return std::make_pair(crop_img, affine_matrix);

}

Ort::Value Face_Restoration::transform(const cv::Mat &mat_rs) {
    input_node_dims[0] = 1;
    input_node_dims[1] = mat_rs.channels();
    input_node_dims[2] = mat_rs.rows;
    input_node_dims[3] = mat_rs.cols;

    return ortcv::utils::transform::create_tensor(
            mat_rs, input_node_dims, memory_info_handler,
            input_values_handler, ortcv::utils::transform::CHW);
}

cv::Mat blend_frame(const cv::Mat &target_image, const cv::Mat &paste_frame)
{
    float face_enhancer_blend = 1.0f - (80.0f / 100.0f);  // 计算得到 0.2

    cv::Mat temp_vision_frame;  // 先声明输出Mat

    cv::addWeighted(target_image, face_enhancer_blend,     // 第一个图像和权重
                    paste_frame, 1.0f - face_enhancer_blend, // 第二个图像和权重
                    0,                                      // gamma值
                    temp_vision_frame);                     // 输出Mat

    return temp_vision_frame;
}


void Face_Restoration::detect(cv::Mat &face_swap_image, std::vector<cv::Point2f > &target_landmarks_5 , const std::string &face_enchaner_path) {
    auto ori_image = face_swap_image.clone();

    cv::Mat crop_image;
    cv::Mat affine_matrix;
    std::tie(crop_image,affine_matrix) = warp_face_by_face_landmark_5(face_swap_image,target_landmarks_5);

    std::vector<float> crop_size = {512,512};
    cv::Mat box_mask = create_static_box_mask(crop_size);
    std::vector<cv::Mat> crop_mask_list;
    crop_mask_list.emplace_back(box_mask);

    cv::cvtColor(crop_image,crop_image,cv::COLOR_BGR2RGB);
    crop_image.convertTo(crop_image,CV_32FC3,1.f / 255.f);
    crop_image.convertTo(crop_image,CV_32FC3,2.0f,-1.f);

    Ort::Value input_tensor = transform(crop_image);

    Ort::RunOptions runOptions;

    // 2.infer
    auto output_tensors = ort_session->Run(
            runOptions, input_node_names.data(),
            &input_tensor, 1, output_node_names.data(), num_outputs
    );

    float *pdata = output_tensors[0].GetTensorMutableData<float>();
    std::vector<int64_t> out_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    int channel = 3;
    int height = 512;
    int width = 512;
    std::vector<float> output(channel * height * width);
    output.assign(pdata,pdata + (channel * height * width));

    std::transform(output.begin(),output.end(),output.begin(),
                   [](double x){return std::max(-1.0,std::max(-1.0,std::min(1.0,x)));});

    std::transform(output.begin(),output.end(),output.begin(),
                   [](double x){return (x + 1.f) /2.f;});


    std::vector<float> transposed_data(channel * height * width);
    for (int c = 0; c < channel; ++c){
        for (int h = 0 ; h < height; ++h){
            for (int w = 0; w < width ; ++w){
                int src_index = c * (height * width) + h * width + w;
                int dst_index = h * (width * channel) + w *  channel + c;
                transposed_data[dst_index] = output[src_index];
            }
        }
    }

    std::transform(transposed_data.begin(),transposed_data.end(),transposed_data.begin(),
                   [](float x){return std::round(x * 255.f);});

    // 1. 先将浮点数转换为uint8
    std::transform(transposed_data.begin(), transposed_data.end(), transposed_data.begin(),
                   [](float x) { return static_cast<uint8_t>(x); });


    cv::Mat mat(height, width, CV_32FC3, transposed_data.data());
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);


    auto crop_mask = crop_mask_list[0];
    cv::Mat paste_frame = paste_back(ori_image,mat,crop_mask,affine_matrix);

    cv::Mat dst_image = blend_frame(ori_image,paste_frame);

    cv::imwrite(face_enchaner_path,dst_image);

}