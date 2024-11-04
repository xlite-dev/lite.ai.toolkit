//
// Created by wangzijian on 11/1/24.
//

#include "face_68landmarks.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/utils.h"

using ortcv::Face_68Landmarks;

// C++版的warp_face_by_translation函数
std::pair<cv::Mat, cv::Mat> warp_face_by_translation(const cv::Mat& temp_img,cv::Point2f& translation, float scale, const cv::Size& crop_size) {
    // 定义仿射矩阵
    cv::Mat affine_matrix = (cv::Mat_<float>(2, 3) << scale, 0, translation.x,
            0, scale, translation.y);

    // 进行仿射变换
    cv::Mat crop_img;
    cv::warpAffine(temp_img, crop_img, affine_matrix, crop_size);

    // 返回裁剪后的图像和仿射矩阵
    return {crop_img, affine_matrix};
}





void Face_68Landmarks::preprocess(const lite::types::Boxf &bounding_box,
                                  const cv::Mat &input_mat,
                                  cv::Mat &crop_img) {
    // 获取边界框的坐标
    float xmin = bounding_box.x1;  // 左上角x
    float ymin = bounding_box.y1;  // 左上角y
    float xmax = bounding_box.x2;  // 右下角x
    float ymax = bounding_box.y2;  // 右下角y

    // 计算缩放因子
    // 找出宽度和高度的最大值
    float width = xmax - xmin;
    float height = ymax - ymin;
    float max_side = std::max(width, height);
    float scale = 195.0f / max_side;

    // 计算平移参数
    // 边界框中心点的坐标
    float center_x = (xmax + xmin) * scale;
    float center_y = (ymax + ymin) * scale;

    // 计算最终的平移量
    cv::Point2f translation;
    translation.x = (256.0f - center_x) * 0.5f;
    translation.y = (256.0f - center_y) * 0.5f;

    cv::Size crop_size(256, 256);

    std::tie(crop_img, affine_matrix) = warp_face_by_translation(input_mat, translation, scale, crop_size);
    crop_img.convertTo(crop_img,CV_32FC3,1 / 255.f);

}


Ort::Value Face_68Landmarks::transform(const cv::Mat &mat_rs) {
    input_node_dims[0] = 1;
    input_node_dims[1] = mat_rs.channels();
    input_node_dims[2] = mat_rs.rows;
    input_node_dims[3] = mat_rs.cols;

    return ortcv::utils::transform::create_tensor(
            mat_rs, input_node_dims, memory_info_handler,
            input_values_handler, ortcv::utils::transform::CHW);
}


void Face_68Landmarks::detect(const cv::Mat &input_mat, const lite::types::BoundingBoxType<float, float> &bbox,
                              const std::string &output_path) {
    if (input_mat.empty()) return;

    img_with_landmarks = input_mat.clone();
    cv::Mat crop_image;

    preprocess(bbox,input_mat,crop_image);

    // 得到crop_image这个变量在ORT中将其转化
    Ort::Value input_tensor = transform(crop_image);
    Ort::RunOptions runOptions;

    // 2.infer
    auto output_tensors = ort_session->Run(
            runOptions, input_node_names.data(),
            &input_tensor, 1, output_node_names.data(), num_outputs
    );

    postprocess(output_tensors,output_path);

}

std::vector<cv::Point2f> convert_face_landmark_68_to_5(const std::vector<cv::Point2f>& landmark_68) {
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




void Face_68Landmarks::postprocess(std::vector<Ort::Value> &ort_outputs, const std::string &output_path) {
    float *pdata = ort_outputs[0].GetTensorMutableData<float>();
    std::vector<int64_t> out_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    std::vector<cv::Point2f> landmarks;
    std::cout<<"test"<<std::endl;
    for (int i = 0;i < 68; ++i)
    {
        // 只取前两个坐标(x,y)并归一化
        float x = pdata[i * 3] / 64.0f  * 256.f;       // 取x坐标
        float y = pdata[i * 3 + 1] / 64.0f * 256.f;   // 取y坐标
        landmarks.emplace_back(x, y);
    }


    // 计算逆仿射变换矩阵
    cv::Mat inverse_affine_matrix;
    cv::invertAffineTransform(affine_matrix, inverse_affine_matrix);
    // 应用逆仿射变换到 face_landmark_68
    cv::transform(landmarks, landmarks, inverse_affine_matrix);
//
//    // 将 68 点关键点转换为 5 点
//    std::vector<cv::Point2f> face_landmark_5of68 = convert_face_landmark_68_to_5(landmarks);
//
    // 绘制每个关键点
    for (const auto& point : landmarks) {
        cv::circle(img_with_landmarks, cv::Point(static_cast<int>(point.x), static_cast<int>(point.y)), 3, cv::Scalar(0, 255, 0), -1);
    }

//    // 保存带关键点的图像
    cv::imwrite(output_path, img_with_landmarks);





}