//
// Created by wangzijian on 2/26/25.
//

#include "docunwarp.h"
#include "lite/ort/core/ort_utils.h"
using ortcv::DocUnWarp;


cv::Mat restore_original_size(const cv::Mat& image, const std::vector<float> &pad_info) {

    if (pad_info.size()  != 4) {
        throw std::invalid_argument("pad_info must contain exactly 4 elements [x,y,h,w]");
    }

    int start_x = static_cast<int>(pad_info[0]);
    int start_y = static_cast<int>(pad_info[1]);
    int original_height = static_cast<int>(pad_info[2]);
    int original_width = static_cast<int>(pad_info[3]);

    // check
    if (start_x < 0 || start_y < 0 ||
        original_width <= 0 || original_height <= 0 ||
        (start_x + original_width) > image.cols  ||
        (start_y + original_height) > image.rows)  {
        throw std::out_of_range("Crop region exceeds image boundaries");
    }

    cv::Rect roi(start_x, start_y, original_width, original_height);

    return image(roi).clone(); // 若需深拷贝则保留.clone()
}


std::tuple<cv::Mat ,std::vector<float> > pad_to_multiple_of_n(const cv::Mat &image,int n =32){
    auto original_height  = image.rows;
    auto original_width = image.cols;
    auto target_width = (int)(((original_width + n - 1) / n) * n);
    auto target_height = (int)(((original_height + n - 1) / n) * n);
    cv::Mat padded_image(target_height, target_width, CV_8UC3, cv::Scalar(255, 255, 255));

    auto start_x = (int)((target_width - original_width) / 2);
    auto start_y = (int)((target_height - original_height) / 2);
    // 将原始图像放置在填充图像的指定位置
    image.copyTo(padded_image(cv::Rect(start_x, start_y, image.cols, image.rows)));

    return std::make_tuple(padded_image, std::vector<float>{(float)start_x,(float)start_y,(float)original_height,(float)original_width});


}

void DocUnWarp::preprocess(const cv::Mat &input_image, cv::Mat &preprocessed_image) {
    cv::Mat temp_image;
    cv::cvtColor(input_image, temp_image,cv::COLOR_BGR2RGB);
    auto result = pad_to_multiple_of_n(temp_image);
    cv::Mat image = std::get<0>(result);
    pad_info = std::get<1>(result);
    image.convertTo(preprocessed_image,CV_32FC3,1/ 255.f,0);
}

void DocUnWarp::postprocess( std::vector<Ort::Value> &pred, cv::Mat &postprocess_mat) {
    float *pdata = pred[0].GetTensorMutableData<float>();
    auto num = pred[0].GetTensorTypeAndShapeInfo().GetShape();

    // 获取输出维度
    int height = num[2];
    int width = num[3];

    // 创建单通道图像
    cv::Mat single_channel(height, width, CV_32FC1);
    memcpy(single_channel.data, pdata, height * width * sizeof(float));

    // 归一化 (1 - (x - min) / (max - min))
    double min_val, max_val;
    cv::minMaxLoc(single_channel, &min_val, &max_val);
    single_channel = 1.0f - (single_channel - min_val) / (max_val - min_val);

    // 从单通道转换为三通道图像（等同于np.repeat(img, 3, axis=2)）
    cv::Mat three_channel;
    cv::cvtColor(single_channel, three_channel, cv::COLOR_GRAY2BGR);

    // 缩放到0-255范围并转换为8位无符号整数
    three_channel = three_channel * 255;
    cv::Mat temp_mat;
    three_channel.convertTo(temp_mat, CV_8UC3);
    postprocess_mat = restore_original_size(temp_mat,pad_info);
}

Ort::Value DocUnWarp::transform(const cv::Mat &mat) {
    std::vector<long> input_node_dims_current = {1,3,mat.rows,mat.cols};
    return ortcv::utils::transform::create_tensor(
            mat, input_node_dims_current, memory_info_handler,
            input_values_handler, ortcv::utils::transform::CHW);
}

void DocUnWarp::detect(const cv::Mat &input_image, cv::Mat &out_image) {
    cv::Mat preprocessed_image;
    preprocess(input_image,preprocessed_image);
    // 1. make input tensor
    Ort::Value input_tensor = transform(preprocessed_image);

    Ort::RunOptions runOptions;

    // 2. inference scores & boxes.
    auto output_tensors = ort_session->Run(
            runOptions, input_node_names.data(),
            &input_tensor, 1, output_node_names.data(), num_outputs
    );

    std::cout<<"infer done!"<<std::endl;

    postprocess(output_tensors,out_image);

}
