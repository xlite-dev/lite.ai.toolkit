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