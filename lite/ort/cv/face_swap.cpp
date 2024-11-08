//
// Created by wangzijian on 11/5/24.
//

#include "face_swap.h"
using ortcv::Face_Swap;

std::vector<float> load_npy(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // 验证magic number
    char magic[6];
    file.read(magic, 6);
    if (magic[0] != '\x93' || magic[1] != 'N' || magic[2] != 'U' ||
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
        throw std::runtime_error("Invalid .npy file format");
    }

    // 读取版本
    uint8_t major_version, minor_version;
    file.read(reinterpret_cast<char*>(&major_version), 1);
    file.read(reinterpret_cast<char*>(&minor_version), 1);

    // 读取header长度
    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);

    // 读取header字符串
    std::vector<char> header(header_len);
    file.read(header.data(), header_len);

    // 解析header获取形状信息（这里简化处理，假设是简单的一维或二维数组）
    size_t num_elements = 512 * 512; // 假设知道数组大小为512

    // 读取数据
    std::vector<float> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(float));

    return data;
}


std::vector<float> dot_product(const std::vector<float>& vec,
                               const std::vector<float>& matrix,
                               int matrix_cols) {
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

float calculate_norm(const std::vector<float>& vec) {
    float sum = 0.0f;
    for (float v : vec) {
        sum += v * v;
    }
    return std::sqrt(sum);
}


void normalize(std::vector<float>& vec) {
    float norm = calculate_norm(vec);
    if (norm > 0) {
        for (float& v : vec) {
            v /= norm;
        }
    }
}


std::pair<cv::Mat, cv::Mat> Face_Swap::warp_face_by_face_landmark_5(cv::Mat input_mat, std::vector<cv::Point2f> face_landmark_5) {
    // 创建标准模板点
    std::vector<cv::Point2f> normed_template;
    for(auto current_template : face_template)  // face_template应该是类的成员变量
    {
        current_template.x = current_template.x * 128;  // 118
        current_template.y = current_template.y * 128;  // 注意：原代码中y使用了x，这里修正为y
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
            cv::Size(128, 128), // 输出大小
            cv::INTER_AREA,     // 插值方法
            cv::BORDER_REPLICATE // 边界处理方式
    );

    // 返回结果对
    return std::make_pair(crop_img, affine_matrix);
}


cv::Mat Face_Swap::create_static_box_mask(std::vector<float> crop_size)
{
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

void Face_Swap::preprocess(cv::Mat &target_face, std::vector<float> source_image_embeding,
                           std::vector<cv::Point2f> target_landmark_5,std::vector<float> &processed_source_embeding,
                           cv::Mat &preprocessed_mat) {

//    cv::Mat preprocessed_mat;

    std::tie(preprocessed_mat, affine_martix) = warp_face_by_face_landmark_5(target_face,target_landmark_5);

    std::vector<float> crop_size= {128.0,128.0};
    crop_list.emplace_back(create_static_box_mask(crop_size));

    cv::cvtColor(preprocessed_mat,preprocessed_mat,cv::COLOR_BGR2RGB);
    preprocessed_mat.convertTo(preprocessed_mat,CV_32FC3,1.0 / 255.f);
    preprocessed_mat.convertTo(preprocessed_mat,CV_32FC3,1.0 / 1.f,0);

    std::vector<float> model_martix = load_npy("/home/facefusion-onnxrun/python/model_matrix.npy");

    processed_source_embeding= dot_product(source_image_embeding,model_martix,512);


    normalize(processed_source_embeding);

    std::cout<<"done!"<<std::endl;
}


Ort::Value Face_Swap::transform(const cv::Mat &mat_rs) {
    input_node_dims[0] = 1;
    input_node_dims[1] = mat_rs.channels();
    input_node_dims[2] = mat_rs.rows;
    input_node_dims[3] = mat_rs.cols;

    return ortcv::utils::transform::create_tensor(
            mat_rs, input_node_dims, memory_info_handler,
            input_values_handler, ortcv::utils::transform::CHW);
}

void Face_Swap::postprocess(cv::Mat &input_mat, std::string &output_path) {

}


cv::Mat paste_back(const cv::Mat& temp_vision_frame,
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


void Face_Swap::detect(cv::Mat &target_image,std::vector<float> source_face_embeding,std::vector<cv::Point2f> target_landmark_5,
                       cv::Mat &face_swap_image){

    cv::Mat ori_image = target_image.clone();
    std::vector<float> source_embeding_input;
    cv::Mat model_input_mat;
    preprocess(target_image,source_face_embeding,target_landmark_5,source_embeding_input,model_input_mat);
    Ort::Value inputTensor_target = transform(model_input_mat);

    std::vector<int64_t> input_node_dims = {1, 512};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor_embeding = Ort::Value::CreateTensor<float>(
            memory_info,
            source_embeding_input.data(),
            source_embeding_input.size(),
            input_node_dims.data(),
            input_node_dims.size()
    );

    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(std::move(inputTensor_target));
    inputTensors.push_back(std::move(inputTensor_embeding));


    Ort::RunOptions runOptions;

    std::vector<const char *> input_node_names_face_swap = {
            "target",
            "source",
    };

    std::vector<const char *> output_node_names_face_swap = {
            "output"
    };

    std::vector<Ort::Value> outputTensors = ort_session->Run(
            runOptions,
            input_node_names_face_swap.data(),
            inputTensors.data(),
            inputTensors.size(),
            output_node_names_face_swap.data(),
            output_node_names_face_swap.size()
    );

    float *p_data = outputTensors[0].GetTensorMutableData<float>();
    std::vector<int64_t> out_shape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

    std::vector<float> output_swap_image(1 * 3 * 128 * 128);
    output_swap_image.assign(p_data,p_data + (1 * 3 * 128 * 128));

    std::vector<float> transposed(3 * 128 * 128);
    int channels = 3;
    int height = 128;
    int width = 128;

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int src_idx = c * (height * width) + h * width + w;  // CHW
                int dst_idx = h * (width * channels) + w * channels + c;  // HWC
                transposed[dst_idx] = output_swap_image[src_idx];
            }
        }
    }

    for (auto& val : transposed) {
        val = std::round(val * 255.0);
    }

    // 3. 使用OpenCV进行颜色通道转换
    cv::Mat mat(height, width, CV_32FC3, transposed.data());
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);

    cv::Mat dst_image = paste_back(ori_image,mat,crop_list[0],affine_martix);

    face_swap_image = dst_image;


}