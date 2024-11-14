//
// Created by wangzijian on 11/12/24.
//

#include "trt_face_68landmarks.h"
using trtcv::TRTFaceFusionFace68Landmarks;

void
TRTFaceFusionFace68Landmarks::preprocess(const lite::types::Boxf &bounding_box, const cv::Mat &input_mat, cv::Mat &crop_img) {
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

void TRTFaceFusionFace68Landmarks::detect(const cv::Mat &input_mat, const lite::types::BoundingBoxType<float, float> &bbox,
                                 std::vector<cv::Point2f> &face_landmark_5of68) {
    if (input_mat.empty()) return;

    img_with_landmarks = input_mat.clone();
    cv::Mat crop_image;

    preprocess(bbox,input_mat,crop_image);

    std::vector<float> input_data;

    trtcv::utils::transform::create_tensor(crop_image,input_data,input_node_dims,trtcv::utils::transform::CHW);

    cudaMemcpyAsync(buffers[0], input_data.data(), input_node_dims[0] * input_node_dims[1] * input_node_dims[2] * input_node_dims[3] * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // 在推理之前同步流，確保數據完全拷貝
    cudaStreamSynchronize(stream);
    bool status = trt_context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    if (!status){
        std::cerr << "Failed to infer by TensorRT." << std::endl;
        return;
    }

    std::vector<float> output(output_node_dims[0][0] * output_node_dims[0][1] * output_node_dims[0][2]);
    cudaMemcpyAsync(output.data(), buffers[1], output_node_dims[0][0] * output_node_dims[0][1] * output_node_dims[0][2] * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);


    postprocess(output.data(),face_landmark_5of68);

}


void TRTFaceFusionFace68Landmarks::postprocess(float *trt_outputs, std::vector<cv::Point2f> &face_landmark_5of68) {
    std::vector<cv::Point2f> landmarks;

    for (int i = 0;i < 68; ++i)
    {
        float x = trt_outputs[i * 3] / 64.0f  * 256.f;
        float y = trt_outputs[i * 3 + 1] / 64.0f * 256.f;
        landmarks.emplace_back(x, y);
    }

    cv::Mat inverse_affine_matrix;
    cv::invertAffineTransform(affine_matrix, inverse_affine_matrix);

    cv::transform(landmarks, landmarks, inverse_affine_matrix);

    face_landmark_5of68 = face_utils::convert_face_landmark_68_to_5(landmarks);
}


