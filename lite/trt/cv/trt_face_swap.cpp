//
// Created by wangzijian on 11/13/24.
//

#include "trt_face_swap.h"
using trtcv::TRTFaceFusionFaceSwap;

void TRTFaceFusionFaceSwap::preprocess(cv::Mat &target_face, std::vector<float> source_image_embeding,
                                       std::vector<cv::Point2f> target_landmark_5,
                                       std::vector<float> &processed_source_embeding, cv::Mat &preprocessed_mat) {

    std::tie(preprocessed_mat, affine_martix) = face_utils::warp_face_by_face_landmark_5(target_face,target_landmark_5,face_utils::ARCFACE_128_V2);

    std::vector<float> crop_size= {128.0,128.0};
    crop_list.emplace_back(face_utils::create_static_box_mask(crop_size));

    cv::cvtColor(preprocessed_mat,preprocessed_mat,cv::COLOR_BGR2RGB);
    preprocessed_mat.convertTo(preprocessed_mat,CV_32FC3,1.0 / 255.f);
    preprocessed_mat.convertTo(preprocessed_mat,CV_32FC3,1.0 / 1.f,0);

    std::vector<float> model_martix = face_utils::load_npy("/home/facefusion-onnxrun/python/model_matrix.npy");

    processed_source_embeding= face_utils::dot_product(source_image_embeding,model_martix,512);

    face_utils::normalize(processed_source_embeding);

    std::cout<<"done!"<<std::endl;

}


void TRTFaceFusionFaceSwap::detect(cv::Mat &target_image, std::vector<float> source_face_embeding,
                                   std::vector<cv::Point2f> target_landmark_5, cv::Mat &face_swap_image) {
    cv::Mat ori_image = target_image.clone();
    std::vector<float> source_embeding_input;
    cv::Mat model_input_mat;
    // 预处理时间
    auto start_preprocess = std::chrono::high_resolution_clock::now();
    preprocess(target_image,source_face_embeding,target_landmark_5,source_embeding_input,model_input_mat);
    auto end_preprocess = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_preprocess = end_preprocess-start_preprocess;
    std::cout << "Face_Swap preprocess Time: " << diff_preprocess.count() * 1000 << " ms\n";

    std::vector<float> input_vector;
    trtcv::utils::transform::create_tensor(model_input_mat,input_vector,input_node_dims,trtcv::utils::transform::CHW);

    // 这个是 source 的输入下面写一个 embeding 的输入
    cudaMemcpyAsync(buffers[0],input_vector.data(),1 * 3 * 128 * 128 *sizeof(float ), cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(buffers[1],source_embeding_input.data(),512 * sizeof(float), cudaMemcpyHostToDevice,stream);

    // 推理之前先同步一下
    cudaStreamSynchronize(stream);

    // 这里是推理
    bool status = trt_context->enqueueV3(stream);
    if (!status) {
        std::cerr << "Failed to enqueue TensorRT model." << std::endl;
        return;
    }
    auto start = std::chrono::high_resolution_clock::now();

//     将输出拷贝出来
    std::vector<float> output_vector(3 * 128 * 128);
    cudaMemcpyAsync(output_vector.data(),buffers[2],1 * 3 * 128 * 128 * sizeof(float),cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);

    std::vector<float> output_swap_image(1 * 3 * 128 * 128);
    output_swap_image.assign(output_vector.begin(),output_vector.end());



    std::vector<float> transposed(3 * 128 * 128);
    int channels = 3;
    int height = 128;
    int width = 128;
//    launch_face_swap_postprocess(
//            static_cast<float*>(buffers[2]),
//            channels,
//            height,
//            width,
//            transposed.data()
//            );

    // 写一个测试时间的代码

#pragma omp parallel for collapse(3)
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int src_idx = c * (height * width) + h * width + w;  // CHW
                int dst_idx = h * (width * channels) + w * channels + c;  // HWC
                transposed[dst_idx] = output_swap_image[src_idx];
            }
        }
    }

//    for (int c = 0; c < channels; ++c) {
//        for (int h = 0; h < height; ++h) {
//            for (int w = 0; w < width; ++w) {
//                int src_idx = c * (height * width) + h * width + w;  // CHW
//                int dst_idx = h * (width * channels) + w * channels + c;  // HWC
//                transposed[dst_idx] = output_swap_image[src_idx];
//            }
//        }
//    }

    for (auto& val : transposed) {
        val = std::round(val * 255.0);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "Face_Swap postprocess  Time: " << diff.count() * 1000 << " ms\n";


    cv::Mat mat(height, width, CV_32FC3, transposed.data());
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);

    // 计算pasteback时间
    auto start_pasteback = std::chrono::high_resolution_clock::now();
//    cv::Mat dst_image = face_utils::paste_back(ori_image,mat,crop_list[0],affine_martix);
    cv::Mat dst_image = launch_paste_back(ori_image,mat,crop_list[0],affine_martix);
    auto end_pasteback = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_pasteback = end_pasteback-start_pasteback;
    std::cout << "Face_Swap pasteback Time: " << diff_pasteback.count() * 1000 << " ms\n";
    face_swap_image = dst_image;
}