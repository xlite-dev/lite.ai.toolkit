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



