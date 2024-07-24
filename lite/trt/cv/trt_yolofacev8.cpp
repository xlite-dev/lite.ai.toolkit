//
// Created by ai-test1 on 24-7-11.
//

#include "trt_yolofacev8.h"
using trtcv::TRTYoloFaceV8;

float TRTYoloFaceV8::get_iou(const lite::types::Boxf box1, const lite::types::Boxf box2) {
    float x1 = std::max(box1.x1, box2.x1);
    float y1 = std::max(box1.y1, box2.y1);
    float x2 = std::min(box1.x2, box2.x2);
    float y2 = std::min(box1.y2, box2.y2);
    float w = std::max(0.f, x2 - x1);
    float h = std::max(0.f, y2 - y1);
    float over_area = w * h;
    if (over_area == 0)
        return 0.0;
    float union_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1) + (box2.x2 - box2.x1) * (box2.y2 - box2.y1) - over_area;
    return over_area / union_area;
}


std::vector<int> TRTYoloFaceV8::nms(std::vector<lite::types::Boxf> boxes, std::vector<float> confidences, const float nms_thresh) {
    sort(confidences.begin(), confidences.end(), [&confidences](size_t index_1, size_t index_2)
    { return confidences[index_1] > confidences[index_2]; });
    const int num_box = confidences.size();
    std::vector<bool> isSuppressed(num_box, false);
    for (int i = 0; i < num_box; ++i)
    {
        if (isSuppressed[i])
        {
            continue;
        }
        for (int j = i + 1; j < num_box; ++j)
        {
            if (isSuppressed[j])
            {
                continue;
            }

            float ovr = this->get_iou(boxes[i], boxes[j]);
            if (ovr > nms_thresh)
            {
                isSuppressed[j] = true;
            }
        }
    }

    std::vector<int> keep_inds;
    for (int i = 0; i < isSuppressed.size(); i++)
    {
        if (!isSuppressed[i])
        {
            keep_inds.emplace_back(i);
        }
    }
    return keep_inds;
}

cv::Mat TRTYoloFaceV8::normalize(cv::Mat srcimg) {
    const int height = srcimg.rows;
    const int width = srcimg.cols;
    cv::Mat temp_image = srcimg.clone();
    int input_height = input_node_dims[2];
    int input_width = input_node_dims[3];

    if (height > input_height || width > input_width)
    {
        const float scale = std::min((float)input_height / height, (float)input_width / width);
        cv::Size new_size = cv::Size(int(width * scale), int(height * scale));
        cv::resize(srcimg, temp_image, new_size);
    }

    ratio_height = (float)height / temp_image.rows;
    ratio_width = (float)width / temp_image.cols;

    cv::Mat input_img;
    cv::copyMakeBorder(temp_image, input_img, 0, input_height - temp_image.rows,
                       0, input_width - temp_image.cols, cv::BORDER_CONSTANT, 0);

    std::vector<cv::Mat> bgrChannels(3);
    cv::split(input_img, bgrChannels);
    for (int c = 0; c < 3; c++)
    {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / 128.0, -127.5 / 128.0);
    }
    cv::Mat normalized_image;
    cv::merge(bgrChannels,normalized_image);
    return normalized_image;

}



void TRTYoloFaceV8::generate_box(float *trt_outputs, std::vector<lite::types::Boxf> &boxes, float conf_threshold,
                                 float iou_threshold) {

    int num_box = output_node_dims[0][2];
    std::vector<lite::types::BoundingBoxType<float, float>> bounding_box_raw;
    std::vector<float> score_raw;
    for (int i = 0; i < num_box; i++)
    {
        const float score = trt_outputs[4 * num_box + i];
        if (score > conf_threshold)
        {
            float x1 = (trt_outputs[i] - 0.5 * trt_outputs[2 * num_box + i]) * ratio_width;
            float y1 = (trt_outputs[num_box + i] - 0.5 * trt_outputs[3 * num_box + i]) * ratio_height;
            float x2 = (trt_outputs[i] + 0.5 * trt_outputs[2 * num_box + i]) * ratio_width;
            float y2 = (trt_outputs[num_box + i] + 0.5 * trt_outputs[3 * num_box + i]) * ratio_height;

            lite::types::BoundingBoxType<float, float> bbox;
            bbox.x1 = x1;
            bbox.y1 = y1;
            bbox.x2 = x2;
            bbox.y2 = y2;
            bbox.score = score;
            bbox.flag = true;
            bounding_box_raw.emplace_back(bbox);
            score_raw.emplace_back(score);
        }
    }
    std::vector<int> keep_inds = this->nms(bounding_box_raw, score_raw, iou_threshold);
    const int keep_num = keep_inds.size();
    boxes.clear();
    boxes.resize(keep_num);
    for (int i = 0; i < keep_num; i++)
    {
        const int ind = keep_inds[i];
        boxes[i] = bounding_box_raw[ind];
    }

}


void TRTYoloFaceV8::detect(const cv::Mat &mat, std::vector<lite::types::Boxf> &boxes, float conf_threshold,
                           float iou_threshold) {


    // 1.normalized the input
    cv::Mat normalized_image = normalize(mat);

    // 2.trans to input vector
    std::vector<float> input;
    trtcv::utils::transform::create_tensor(normalized_image,input,input_node_dims,trtcv::utils::transform::CHW);

    // 3. infer
    cudaMemcpyAsync(buffers[0], input.data(), input_node_dims[0] * input_node_dims[1] * input_node_dims[2] * input_node_dims[3] * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    bool status = trt_context->enqueueV3(stream);


    if (!status){
        std::cerr << "Failed to infer by TensorRT." << std::endl;
        return;
    }

    std::vector<float> output(output_node_dims[0][0] * output_node_dims[0][1] * output_node_dims[0][2]);

    cudaMemcpyAsync(output.data(), buffers[1], output_node_dims[0][0] * output_node_dims[0][1] * output_node_dims[0][2] * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    // 4. generate box
    generate_box(output.data(),boxes,0.45f,0.5f);


}
