//
// Created by wangzijian on 11/5/24.
//

#include "face_swap.h"
using ortcv::Face_Swap;

void Face_Swap::preprocess(cv::Mat &target_face, std::vector<float> source_image_embeding,
                           std::vector<cv::Point2f> target_landmark_5,std::vector<float> &processed_source_embeding,
                           cv::Mat &preprocessed_mat) {


    std::tie(preprocessed_mat, affine_martix) = face_utils::warp_face_by_face_landmark_5(target_face,target_landmark_5,face_utils::ARCFACE_128_V2);

    std::vector<float> crop_size= {128.0,128.0};
    crop_list.emplace_back(face_utils::create_static_box_mask(crop_size));

    cv::cvtColor(preprocessed_mat,preprocessed_mat,cv::COLOR_BGR2RGB);
    preprocessed_mat.convertTo(preprocessed_mat,CV_32FC3,1.0 / 255.f);
    preprocessed_mat.convertTo(preprocessed_mat,CV_32FC3,1.0 / 1.f,0);

    // 使用 CMake 传递的 SOURCE_PATH 宏
    std::string model_matrix_path = std::string(SOURCE_PATH) + "/examples/lite/resources/model_matrix.npy";
    std::vector<float> model_martix = face_utils::load_npy(model_matrix_path);

    processed_source_embeding= face_utils::dot_product(source_image_embeding,model_martix,512);

    face_utils::normalize(processed_source_embeding);

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

    cv::Mat mat(height, width, CV_32FC3, transposed.data());
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);

    cv::Mat dst_image = face_utils::paste_back(ori_image,mat,crop_list[0],affine_martix);
    face_swap_image = dst_image;

}