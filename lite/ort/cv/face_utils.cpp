//
// Created by wangzijian on 11/11/24.
//

#include "face_utils.h"

cv::Mat
face_utils::paste_back(const cv::Mat &temp_vision_frame, const cv::Mat &crop_vision_frame, const cv::Mat &crop_mask,
                       const cv::Mat &affine_matrix) {

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

namespace face_utils
{
    const std::vector<cv::Point2f> face_template_128 = {
            cv::Point2f(0.36167656, 0.40387734),
            cv::Point2f(0.63696719, 0.40235469),
            cv::Point2f(0.50019687, 0.56044219),
            cv::Point2f(0.38710391, 0.72160547),
            cv::Point2f(0.61507734, 0.72034453)
    };

    const std::vector<cv::Point2f> face_template_112 = {
            cv::Point2f(0.34191607, 0.46157411),
            cv::Point2f(0.65653393, 0.45983393),
            cv::Point2f(0.50022500, 0.64050536),
            cv::Point2f(0.37097589, 0.82469196),
            cv::Point2f(0.63151696, 0.82325089)
    };

    const std::vector<cv::Point2f> face_template_512 = {
            cv::Point2f(0.37691676, 0.46864664),
            cv::Point2f(0.62285697, 0.46912813),
            cv::Point2f(0.50123859, 0.61331904),
            cv::Point2f(0.39308822, 0.72541100),
            cv::Point2f(0.61150205, 0.72490465)
    };

    const std::vector<std::vector<cv::Point2f>> face_template_vector = {face_template_112, face_template_128, face_template_512};

}


std::pair<cv::Mat, cv::Mat>
face_utils::warp_face_by_face_landmark_5(cv::Mat input_mat, std::vector<cv::Point2f> face_landmark_5,
                                         unsigned int type) {

    std::vector<cv::Point2f> current_template_select;
    if (type == face_utils::ARCFACE_112_V2)
    {
        current_template_select = face_utils::face_template_vector[0];
    }

    if (type == face_utils::ARCFACE_128_V2)
    {
        current_template_select = face_utils::face_template_vector[1];
    }

    if (type == face_utils::FFHQ_512)
    {
        current_template_select = face_utils::face_template_vector[2];
    }

    // 创建标准模板点
    std::vector<cv::Point2f> normed_template;
    for(auto current_template : current_template_select)  // face_template应该是类的成员变量
    {
        current_template.x = current_template.x * type;  // 512
        current_template.y = current_template.y * type;  // 注意：原代码中y使用了x，这里修正为y
        normed_template.emplace_back(current_template);
    }

    // 估计仿射变换矩阵
    cv::Mat inliers;
    cv::Mat affine_matrix = cv::estimateAffinePartial2D(
            face_landmark_5,
            normed_template,
            inliers,
            cv::RANSAC,
            100
    );

    // 检查变换矩阵是否有效
    if (affine_matrix.empty()) {
        throw std::runtime_error("Failed to estimate affine transformation");
    }

    // 进行仿射变换
    cv::Mat crop_img;
    cv::warpAffine(
            input_mat,
            crop_img,
            affine_matrix,
            cv::Size(type, type),
            cv::INTER_AREA,
            cv::BORDER_REPLICATE
    );

    return std::make_pair(crop_img, affine_matrix);
}


std::vector<float>
face_utils::dot_product(const std::vector<float> &vec, const std::vector<float> &matrix, int matrix_cols) {
    std::vector<float> result(matrix_cols);
    int vec_size = vec.size();

    for (int j = 0; j < matrix_cols; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < vec_size; ++i) {
            sum += vec[i] * matrix[i * matrix_cols + j];
        }
        result[j] = sum;
    }
    return result;
}

float face_utils::calculate_norm(const std::vector<float> &vec) {
    float sum = 0.0f;
    for (float v : vec) {
        sum += v * v;
    }
    return std::sqrt(sum);
}


void face_utils::normalize(std::vector<float> &vec) {
    float norm = calculate_norm(vec);
    if (norm > 0) {
        for (float& v : vec) {
            v /= norm;
        }
    }
}

std::vector<float> face_utils::load_npy(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    char magic[6];
    file.read(magic, 6);
    if (magic[0] != '\x93' || magic[1] != 'N' || magic[2] != 'U' ||
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
        throw std::runtime_error("Invalid .npy file format");
    }

    uint8_t major_version, minor_version;
    file.read(reinterpret_cast<char*>(&major_version), 1);
    file.read(reinterpret_cast<char*>(&minor_version), 1);

    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);

    std::vector<char> header(header_len);
    file.read(header.data(), header_len);

    size_t num_elements = 512 * 512;

    // 读取数据
    std::vector<float> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(float));

    return data;
}

std::pair<cv::Mat, cv::Mat>
face_utils::warp_face_by_translation(const cv::Mat &temp_img, cv::Point2f &translation, float scale,
                                     const cv::Size &crop_size) {
    cv::Mat affine_matrix = (cv::Mat_<float>(2, 3) << scale, 0, translation.x,
            0, scale, translation.y);

    cv::Mat crop_img;
    cv::warpAffine(temp_img, crop_img, affine_matrix, crop_size);

    return {crop_img, affine_matrix};
}


std::vector<cv::Point2f> face_utils::convert_face_landmark_68_to_5(const std::vector<cv::Point2f> &landmark_68) {
    std::vector<cv::Point2f> face_landmark_5;

    // 计算左眼的中心位置
    cv::Point2f left_eye(0.0f, 0.0f);
    for (int i = 36; i < 42; ++i) {
        left_eye += landmark_68[i];
    }
    left_eye *= (1.0f / 6.0f); // 取平均

    // 计算右眼的中心位置
    cv::Point2f right_eye(0.0f, 0.0f);
    for (int i = 42; i < 48; ++i) {
        right_eye += landmark_68[i];
    }
    right_eye *= (1.0f / 6.0f); // 取平均

    // 获取鼻尖位置
    cv::Point2f nose = landmark_68[30];

    // 获取左右嘴角的位置
    cv::Point2f left_mouth_end = landmark_68[48];
    cv::Point2f right_mouth_end = landmark_68[54];

    // 将5个点加入到结果中
    face_landmark_5.push_back(left_eye);
    face_landmark_5.push_back(right_eye);
    face_landmark_5.push_back(nose);
    face_landmark_5.push_back(left_mouth_end);
    face_landmark_5.push_back(right_mouth_end);

    return face_landmark_5;
}

cv::Mat face_utils::blend_frame(const cv::Mat &target_image, const cv::Mat &paste_frame) {
    float face_enhancer_blend = 1.0f - (80.0f / 100.0f);

    cv::Mat temp_vision_frame;

    cv::addWeighted(target_image, face_enhancer_blend,
                    paste_frame, 1.0f - face_enhancer_blend,
                    0,
                    temp_vision_frame);

    return temp_vision_frame;
}


cv::Mat face_utils::create_static_box_mask(std::vector<float> crop_size) {

    float face_mask_blur = 0.3;

    std::vector<int> face_mask_padding = {0,0,0,0};

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
