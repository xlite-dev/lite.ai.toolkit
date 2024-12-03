//
// Created by wangzijian on 11/14/24.
//

#include "trt_face_restoration.h"
using trtcv::TRTFaceFusionFaceRestoration;

void TRTFaceFusionFaceRestoration::detect(cv::Mat &face_swap_image, std::vector<cv::Point2f> &target_landmarks_5,
                                          const std::string &face_enchaner_path) {
    auto ori_image = face_swap_image.clone();

    cv::Mat crop_image;
    cv::Mat affine_matrix;
    std::tie(crop_image,affine_matrix) = face_utils::warp_face_by_face_landmark_5(face_swap_image,target_landmarks_5,face_utils::FFHQ_512);

    std::vector<float> crop_size = {512,512};
    cv::Mat box_mask = face_utils::create_static_box_mask(crop_size);
    std::vector<cv::Mat> crop_mask_list;
    crop_mask_list.emplace_back(box_mask);

    cv::Mat crop_image_rgb;
    launch_bgr2rgb(crop_image,crop_image_rgb);
    crop_image_rgb.convertTo(crop_image_rgb,CV_32FC3,1.f / 255.f);
    crop_image_rgb.convertTo(crop_image_rgb,CV_32FC3,2.0f,-1.f);

    std::vector<float> input_vector;
    trtcv::utils::transform::create_tensor(crop_image_rgb,input_vector,input_node_dims,trtcv::utils::transform::CHW);

    // 拷贝

    // 先不用拷贝了 处理完成再拷贝出来 类似于整个后处理放在GPU上完成
    cudaMemcpyAsync(buffers[0],input_vector.data(),1 * 3 * 512 * 512 * sizeof(float),cudaMemcpyHostToDevice,stream);

    // 同步
    cudaStreamSynchronize(stream);

    // 推理
    bool status = trt_context->enqueueV3(stream);
    if (!status) {
        std::cerr << "Failed to inference" << std::endl;
        return;
    }


    // 同步
    cudaStreamSynchronize(stream);
    std::vector<unsigned char> transposed_data(1 * 3 * 512 * 512);

//    std::vector<float> transposed_data(1 * 3 * 512 * 512);

    // 这里buffer1就是输出了
    launch_face_restoration_postprocess(
            static_cast<float*>(buffers[1]),
            transposed_data.data(),
            3,
            512,
            512
            );

    std::vector<float> transposed_data_float(transposed_data.begin(),
                                             transposed_data.end());


    // 获取输出
    std::vector<float> output_vector(1 * 3 * 512 * 512);
//    cudaMemcpyAsync(output_vector.data(),buffers[1],1 * 3 * 512 * 512 * sizeof(float),cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);
//
    // 后处理
    int channel = 3;
    int height = 512;
    int width = 512;


    cv::Mat mat(height, width, CV_32FC3, transposed_data_float.data());
//    cv::imwrite("/home/lite.ai.toolkit/mid_process.jpg",mat);
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);


    auto crop_mask = crop_mask_list[0];
    cv::Mat paste_frame = face_utils::paste_back(ori_image,mat,crop_mask,affine_matrix);

    cv::Mat dst_image = face_utils::blend_frame(ori_image,paste_frame);

    cv::imwrite(face_enchaner_path,dst_image);
}