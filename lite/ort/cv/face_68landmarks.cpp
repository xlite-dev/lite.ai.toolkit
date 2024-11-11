//
// Created by wangzijian on 11/1/24.
//

#include "face_68landmarks.h"

using ortcv::Face_68Landmarks;

void Face_68Landmarks::preprocess(const lite::types::Boxf &bounding_box,
                                  const cv::Mat &input_mat,
                                  cv::Mat &crop_img) {

    float xmin = bounding_box.x1;
    float ymin = bounding_box.y1;
    float xmax = bounding_box.x2;
    float ymax = bounding_box.y2;


    float width = xmax - xmin;
    float height = ymax - ymin;
    float max_side = std::max(width, height);
    float scale = 195.0f / max_side;

    float center_x = (xmax + xmin) * scale;
    float center_y = (ymax + ymin) * scale;

    cv::Point2f translation;
    translation.x = (256.0f - center_x) * 0.5f;
    translation.y = (256.0f - center_y) * 0.5f;

    cv::Size crop_size(256, 256);

    std::tie(crop_img, affine_matrix) = face_utils::warp_face_by_translation(input_mat, translation, scale, crop_size);

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
                               std::vector<cv::Point2f> &face_landmark_5of68) {
    if (input_mat.empty()) return;

    img_with_landmarks = input_mat.clone();
    cv::Mat crop_image;

    preprocess(bbox,input_mat,crop_image);

    Ort::Value input_tensor = transform(crop_image);
    Ort::RunOptions runOptions;

    // 2.infer
    auto output_tensors = ort_session->Run(
            runOptions, input_node_names.data(),
            &input_tensor, 1, output_node_names.data(), num_outputs
    );

    postprocess(output_tensors,face_landmark_5of68);

}



void Face_68Landmarks::postprocess(std::vector<Ort::Value> &ort_outputs,
                                   std::vector<cv::Point2f> &face_landmark_5of68) {
    float *pdata = ort_outputs[0].GetTensorMutableData<float>();
    std::vector<int64_t> out_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    std::vector<cv::Point2f> landmarks;

    for (int i = 0;i < 68; ++i)
    {
        float x = pdata[i * 3] / 64.0f  * 256.f;
        float y = pdata[i * 3 + 1] / 64.0f * 256.f;
        landmarks.emplace_back(x, y);
    }

    cv::Mat inverse_affine_matrix;
    cv::invertAffineTransform(affine_matrix, inverse_affine_matrix);

    cv::transform(landmarks, landmarks, inverse_affine_matrix);

    face_landmark_5of68 = face_utils::convert_face_landmark_68_to_5(landmarks);
}


