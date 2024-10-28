//
// Created by ai-test1 on 24-7-11.
//

#include "trt_utils.h"


void trtcv::utils::transform::create_tensor(const cv::Mat &mat,std::vector<float> &input_vector,std::vector<int64_t> input_node_dims,unsigned int data_format){
    // make mat to float type's vector

    const unsigned int rows = mat.rows;
    const unsigned int cols = mat.cols;
    const unsigned int channels = mat.channels();

    cv::Mat mat_ref;
    if (mat.type() != CV_32FC(channels)) mat.convertTo(mat_ref, CV_32FC(channels));
    else mat_ref = mat; // reference only. zero-time cost. support 1/2/3/... channels

    if (input_node_dims.size() != 4) throw std::runtime_error("dims mismatch.");
    if (input_node_dims.at(0) != 1) throw std::runtime_error("batch != 1");

    if (data_format == transform::CHW)
    {
        const unsigned int target_tensor_size = rows * cols * channels;
        // input vector's size
        input_vector.resize(target_tensor_size);

        for (int c = 0; c < channels; ++c)
        {
            for (int h = 0; h < rows; ++h)
            {
                for (int w = 0; w < cols; ++w)
                {
                    input_vector[c * rows * cols + h * cols + w] = mat.at<cv::Vec3f>(h, w)[c];
                }
            }
        }

    }else
    {
        throw std::runtime_error("data_format must be transform::CHW!");
    }

}


std::vector<float> trtcv::utils::transform::trt_load_from_bin(const std::string &filename) {
    std::ifstream infile(filename, std::ios::in | std::ios::binary);
    std::vector<float> data;

    if (infile.is_open()) {
        infile.seekg(0, std::ios::end);
        size_t size = infile.tellg();
        infile.seekg(0, std::ios::beg);

        data.resize(size / sizeof(float));
        infile.read(reinterpret_cast<char*>(data.data()), size);
        infile.close();
    } else {
        std::cerr << "Failed to open file: " << filename << std::endl;
    }

    return data;
}

void trtcv::utils::transform::trt_save_to_bin(const std::vector<float> &data, const std::string &filename) {
    std::ofstream outfile(filename, std::ios::out | std::ios::binary);
    if (outfile.is_open()) {
        outfile.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
        outfile.close();
    } else {
        std::cerr << "Failed to open file: " << filename << std::endl;
    }
}

void trtcv::utils::transform::trt_generate_latents(std::vector<float> &latents, int batch_size, int unet_channels,
                                                   int latent_height, int latent_width, float init_noise_sigma) {
    size_t total_size = batch_size * unet_channels * latent_height * latent_width;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < total_size; ++i) {
        latents[i] = dist(gen) * init_noise_sigma;
    }
}

void trtcv::utils::remove_small_connected_area(cv::Mat &alpha_pred, float threshold) {
    cv::Mat gray, binary;
    alpha_pred.convertTo(gray, CV_8UC1, 255.f);
    // 255 * 0.05 ~ 13
    unsigned int binary_threshold = (unsigned int) (255.f * threshold);
    // https://github.com/yucornetto/MGMatting/blob/main/code-base/utils/util.py#L209
    cv::threshold(gray, binary, binary_threshold, 255, cv::THRESH_BINARY);
    // morphologyEx with OPEN operation to remove noise first.
    auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
    // Computationally connected domain
    cv::Mat labels = cv::Mat::zeros(alpha_pred.size(), CV_32S);
    cv::Mat stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, 4);
    if (num_labels <= 1) return; // no noise, skip.
    // find max connected area, 0 is background
    int max_connected_id = 1; // 1,2,...
    int max_connected_area = stats.at<int>(max_connected_id, cv::CC_STAT_AREA);
    for (int i = 1; i < num_labels; ++i)
    {
        int tmp_connected_area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (tmp_connected_area > max_connected_area)
        {
            max_connected_area = tmp_connected_area;
            max_connected_id = i;
        }
    }
    const int h = alpha_pred.rows;
    const int w = alpha_pred.cols;
    // remove small connected area.
    for (int i = 0; i < h; ++i)
    {
        int *label_row_ptr = labels.ptr<int>(i);
        float *alpha_row_ptr = alpha_pred.ptr<float>(i);
        for (int j = 0; j < w; ++j)
        {
            if (label_row_ptr[j] != max_connected_id)
                alpha_row_ptr[j] = 0.f;
        }
    }
}