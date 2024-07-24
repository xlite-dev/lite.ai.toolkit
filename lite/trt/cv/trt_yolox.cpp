//
// Created by wangzijian on 7/22/24.
//

#include "trt_yolox.h"
using trtcv::TRTYoloX;


void TRTYoloX::normalized(cv::Mat &mat_inplace, const float *mean, const float *scale)
{
    // Convert to 32-bit float if not already
    if (mat_inplace.type() != CV_32FC3)
        mat_inplace.convertTo(mat_inplace, CV_32FC3);

    // Split the image into three channels
    std::vector<cv::Mat> channels(3);
    cv::split(mat_inplace, channels);

    // Apply convertTo for each channel
    for (int i = 0; i < 3; ++i)
    {
        // Calculate alpha (scale) and beta (offset)
        float alpha = scale[i];
        float beta = -mean[i] * scale[i];
        channels[i].convertTo(channels[i], CV_32F, alpha, beta);
    }

    // Merge the channels back
    cv::merge(channels, mat_inplace);
}



void TRTYoloX::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs, int target_height, int target_width,
                              trtcv::TRTYoloX::YoloXScaleParams &scale_params) {

    if (mat.empty()) return;
    int img_height = static_cast<int>(mat.rows);
    int img_width = static_cast<int>(mat.cols);

    mat_rs = cv::Mat(target_height, target_width, CV_8UC3,
                     cv::Scalar(114, 114, 114));
    // scale ratio (new / old) new_shape(h,w)
    float w_r = (float) target_width / (float) img_width;
    float h_r = (float) target_height / (float) img_height;
    float r = std::min(w_r, h_r);
    // compute padding
    int new_unpad_w = static_cast<int>((float) img_width * r); // floor
    int new_unpad_h = static_cast<int>((float) img_height * r); // floor
    int pad_w = target_width - new_unpad_w; // >=0
    int pad_h = target_height - new_unpad_h; // >=0

    int dw = pad_w / 2;
    int dh = pad_h / 2;

    // resize with unscaling
    cv::Mat new_unpad_mat;
    // cv::Mat new_unpad_mat = mat.clone(); // may not need clone.
    cv::resize(mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
    new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

    // record scale params.
    scale_params.r = r;
    scale_params.dw = dw;
    scale_params.dh = dh;
    scale_params.new_unpad_w = new_unpad_w;
    scale_params.new_unpad_h = new_unpad_h;
    scale_params.flag = true;
}


void TRTYoloX::generate_anchors(const int target_height,
                             const int target_width,
                             std::vector<int> &strides,
                             std::vector<YoloXAnchor> &anchors)
{
    for (auto stride: strides)
    {
        int num_grid_w = target_width / stride;
        int num_grid_h = target_height / stride;
        for (int g1 = 0; g1 < num_grid_h; ++g1)
        {
            for (int g0 = 0; g0 < num_grid_w; ++g0)
            {
#ifdef LITE_WIN32
        YoloXAnchor anchor;
        anchor.grid0 = g0;
        anchor.grid1 = g1;
        anchor.stride = stride;
        anchors.push_back(anchor);
#else
                anchors.push_back((YoloXAnchor) {g0, g1, stride});
#endif
            }
        }
    }
}


void TRTYoloX::generate_bboxes(const trtcv::TRTYoloX::YoloXScaleParams &scale_params,
                               std::vector<types::Boxf> &bbox_collection, float *output, float score_threshold,
                               int img_height, int img_width) {


    auto pred_dims = output_node_dims[0];
    const unsigned int num_anchors = pred_dims.at(1); // n = ?
    const unsigned int num_classes = pred_dims.at(2) - 5;
    const float input_height = static_cast<float>(input_node_dims.at(2)); // e.g 640
    const float input_width = static_cast<float>(input_node_dims.at(3)); // e.g 640

    std::vector<YoloXAnchor> anchors;
    std::vector<int> strides = {8, 16, 32}; // might have stride=64
    this->generate_anchors(input_height, input_width, strides, anchors);

    float r_ = scale_params.r;
    int dw_ = scale_params.dw;
    int dh_ = scale_params.dh;

    bbox_collection.clear();
    unsigned int count = 0;
    for (unsigned int i = 0; i < num_anchors; ++i)
    {
        float obj_conf = output[i * pred_dims.at(2) + 4];
        if (obj_conf < score_threshold) continue; // filter first.

        float cls_conf = output[i * pred_dims.at(2) + 5];
        unsigned int label = 0;
        for (unsigned int j = 0; j < num_classes; ++j)
        {
            float tmp_conf = output[i * pred_dims.at(2) + 5 + j];
            if (tmp_conf > cls_conf)
            {
                cls_conf = tmp_conf;
                label = j;
            }
        } // argmax
        float conf = obj_conf * cls_conf; // cls_conf (0.,1.)
        if (conf < score_threshold) continue; // filter

        const int grid0 = anchors.at(i).grid0;
        const int grid1 = anchors.at(i).grid1;
        const int stride = anchors.at(i).stride;

        float dx = output[i * pred_dims.at(2) + 0];
        float dy = output[i * pred_dims.at(2) + 1];
        float dw = output[i * pred_dims.at(2) + 2];
        float dh = output[i * pred_dims.at(2) + 3];

        float cx = (dx + (float) grid0) * (float) stride;
        float cy = (dy + (float) grid1) * (float) stride;
        float w = std::exp(dw) * (float) stride;
        float h = std::exp(dh) * (float) stride;
        float x1 = ((cx - w / 2.f) - (float) dw_) / r_;
        float y1 = ((cy - h / 2.f) - (float) dh_) / r_;
        float x2 = ((cx + w / 2.f) - (float) dw_) / r_;
        float y2 = ((cy + h / 2.f) - (float) dh_) / r_;

        types::Boxf box;
        box.x1 = std::max(0.f, x1);
        box.y1 = std::max(0.f, y1);
        box.x2 = std::min(x2, (float) img_width - 1.f);
        box.y2 = std::min(y2, (float) img_height - 1.f);
        box.score = conf;
        box.label = label;
        box.label_text = class_names[label];
        box.flag = true;
        bbox_collection.push_back(box);

        count += 1; // limit boxes for nms.
        if (count > max_nms)
            break;
    }
#if LITETRT_DEBUG
    std::cout << "detected num_anchors: " << num_anchors << "\n";
    std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
#endif

}


void TRTYoloX::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output, float iou_threshold, unsigned int topk,
              unsigned int nms_type) {
    if (nms_type == NMS::BLEND) lite::utils::blending_nms(input, output, iou_threshold, topk);
    else if (nms_type == NMS::OFFSET) lite::utils::offset_nms(input, output, iou_threshold, topk);
    else lite::utils::hard_nms(input, output, iou_threshold, topk);
}


void TRTYoloX::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes, float score_threshold,
                      float iou_threshold, unsigned int topk, unsigned int nms_type) {
    if (mat.empty()) return;
    const int input_height = input_node_dims.at(2);
    const int input_width = input_node_dims.at(3);
    int img_height = static_cast<int>(mat.rows);
    int img_width = static_cast<int>(mat.cols);

    // resize & unscale
    cv::Mat mat_rs;
    YoloXScaleParams scale_params;
    resize_unscale(mat, mat_rs, input_height, input_width, scale_params);

    // normalized image
    normalized(mat_rs,mean_vals,scale_vals);

    //1. make the input
    std::vector<float> input;
    trtcv::utils::transform::create_tensor(mat_rs,input,input_node_dims,trtcv::utils::transform::CHW);

    //2. infer
    cudaMemcpyAsync(buffers[0], input.data(), input_node_dims[0] * input_node_dims[1] * input_node_dims[2] * input_node_dims[3] * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    cudaStreamSynchronize(stream);



    bool status = trt_context->enqueueV3(stream);
    cudaStreamSynchronize(stream);
    if (!status){
        std::cerr << "Failed to infer by TensorRT." << std::endl;
        return;
    }

    cudaStreamSynchronize(stream);


    // get the first output dim
    auto pred_dims = output_node_dims[0];

    std::vector<float> output(pred_dims[0] * pred_dims[1] * pred_dims[2]);

    cudaMemcpyAsync(output.data(), buffers[1], pred_dims[0] * pred_dims[1] * pred_dims[2] * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    //3. generate the boxes
    std::vector<types::Boxf> bbox_collection;
    generate_bboxes(scale_params, bbox_collection, output.data(), score_threshold, img_height, img_width);
    nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);

}