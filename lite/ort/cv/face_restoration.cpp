//
// Created by wangzijian on 11/7/24.
//

#include "face_restoration.h"

using ortcv::Face_Restoration;



Ort::Value Face_Restoration::transform(const cv::Mat &mat_rs) {
    input_node_dims[0] = 1;
    input_node_dims[1] = mat_rs.channels();
    input_node_dims[2] = mat_rs.rows;
    input_node_dims[3] = mat_rs.cols;

    return ortcv::utils::transform::create_tensor(
            mat_rs, input_node_dims, memory_info_handler,
            input_values_handler, ortcv::utils::transform::CHW);
}



void Face_Restoration::detect(cv::Mat &face_swap_image, std::vector<cv::Point2f > &target_landmarks_5 , const std::string &face_enchaner_path) {
    auto ori_image = face_swap_image.clone();

    cv::Mat crop_image;
    cv::Mat affine_matrix;
    std::tie(crop_image,affine_matrix) = face_utils::warp_face_by_face_landmark_5(face_swap_image,target_landmarks_5,face_utils::FFHQ_512);

    std::vector<float> crop_size = {512,512};
    cv::Mat box_mask = face_utils::create_static_box_mask(crop_size);
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

    std::transform(transposed_data.begin(), transposed_data.end(), transposed_data.begin(),
                   [](float x) { return static_cast<uint8_t>(x); });


    cv::Mat mat(height, width, CV_32FC3, transposed_data.data());
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);


    auto crop_mask = crop_mask_list[0];
    cv::Mat paste_frame = face_utils::paste_back(ori_image,mat,crop_mask,affine_matrix);

    cv::Mat dst_image = face_utils::blend_frame(ori_image,paste_frame);

    cv::imwrite(face_enchaner_path,dst_image);

}