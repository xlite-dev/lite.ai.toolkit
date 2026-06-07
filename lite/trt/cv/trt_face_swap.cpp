//
// Created by wangzijian on 11/13/24.
//

#include "trt_face_swap.h"
using trtcv::TRTFaceFusionFaceSwap;

void TRTFaceFusionFaceSwap::preprocess(cv::Mat &target_face, std::vector<float> source_image_embeding,
                                       std::vector<cv::Point2f> target_landmark_5,
                                       std::vector<float> &processed_source_embeding, cv::Mat &preprocessed_mat) {

    std::tie(preprocessed_mat, affine_martix) = face_utils::warp_face_by_face_landmark_5(target_face,target_landmark_5,face_utils::ARCFACE_128_V2);

    cv::cvtColor(preprocessed_mat,preprocessed_mat,cv::COLOR_BGR2RGB);
    preprocessed_mat.convertTo(preprocessed_mat,CV_32FC3,1.0 / 255.f);
    preprocessed_mat.convertTo(preprocessed_mat,CV_32FC3,1.0 / 1.f,0);

    // model_matrix_ and box_mask_ are loaded/built once in the constructor (they are
    // constant); they used to be load_npy'd from disk and rebuilt every frame.
    processed_source_embeding = face_utils::dot_product(source_image_embeding, model_matrix_, 512);
    face_utils::normalize(processed_source_embeding);
}


void TRTFaceFusionFaceSwap::detect(cv::Mat &target_image, std::vector<float> source_face_embeding,
                                   std::vector<cv::Point2f> target_landmark_5, cv::Mat &face_swap_image) {
    cv::Mat ori_image = target_image.clone();
    std::vector<float> source_embeding_input;
    cv::Mat model_input_mat;
    preprocess(target_image,source_face_embeding,target_landmark_5,source_embeding_input,model_input_mat);

    std::vector<float> input_vector;
    trtcv::utils::transform::create_tensor(model_input_mat,input_vector,input_node_dims,trtcv::utils::transform::CHW);

    // input 0 = preprocessed target face, input 1 = source embedding
    cudaMemcpyAsync(buffers[0],input_vector.data(),1 * 3 * 128 * 128 *sizeof(float ), cudaMemcpyHostToDevice,stream);
    cudaMemcpyAsync(buffers[1],source_embeding_input.data(),512 * sizeof(float), cudaMemcpyHostToDevice,stream);
    cudaStreamSynchronize(stream);

    bool status = trt_context->enqueueV3(stream);
    if (!status) {
        std::cerr << "Failed to enqueue TensorRT model." << std::endl;
        return;
    }

    std::vector<float> output_vector(3 * 128 * 128);
    cudaMemcpyAsync(output_vector.data(),buffers[2],1 * 3 * 128 * 128 * sizeof(float),cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);

    std::vector<float> output_swap_image(1 * 3 * 128 * 128);
    output_swap_image.assign(output_vector.begin(),output_vector.end());

    // CHW float -> HWC uint8-range (denormalize); paste-back is done on the GPU below.
    std::vector<float> transposed(3 * 128 * 128);
    const int channels = 3, height = 128, width = 128;
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
    for (auto& val : transposed) {
        val = std::round(val * 255.0);
    }

    cv::Mat mat(height, width, CV_32FC3, transposed.data());
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);

    // GPU-fused paste-back (reused device buffers, no per-frame cudaMalloc), reusing the
    // exact kernel restoration uses. Numerically equivalent to launch_paste_back.
    cv::Mat dst_image = paste_back_gpu_.paste_back(ori_image, mat, box_mask_, affine_martix, stream);
    face_swap_image = dst_image;
}
