//
// Created by wangzijian on 7/26/24.
//

#include "trt_yolov6.h"
using trtcv::TRTYoloV6;


void TRTYoloV6::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs, int target_height, int target_width,
                               trtcv::TRTYoloV6::YOLOv6ScaleParams &scale_params) {
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

void TRTYoloV6::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output, float iou_threshold,
                    unsigned int topk, unsigned int nms_type) {
    if (nms_type == NMS::BLEND) lite::utils::blending_nms(input, output, iou_threshold, topk);
    else if (nms_type == NMS::OFFSET) lite::utils::offset_nms(input, output, iou_threshold, topk);
    else lite::utils::hard_nms(input, output, iou_threshold, topk);
}


void TRTYoloV6::normalized(cv::Mat &input_mat) {
    cv::cvtColor(input_mat,input_mat,cv::COLOR_BGR2RGB);
    input_mat.convertTo(input_mat,CV_32FC3,1.f / 255.f,0.f);
}

void TRTYoloV6::generate_bboxes(const trtcv::TRTYoloV6::YOLOv6ScaleParams &scale_params,
                                std::vector<types::Boxf> &bbox_collection, float *output, float score_threshold,
                                int img_height, int img_width) {
    auto pred_dims = output_node_dims[0];
    const unsigned int num_anchors = pred_dims.at(1); // n = ?
    const unsigned int num_classes = pred_dims.at(2) - 5;

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
        }
        float conf = obj_conf * cls_conf; // cls_conf (0.,1.)
        if (conf < score_threshold) continue; // filter

        float cx = output[i * pred_dims.at(2)];
        float cy = output[i * pred_dims.at(2) + 1];
        float w = output[i * pred_dims.at(2) + 2];
        float h = output[i * pred_dims.at(2) + 3];
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


void TRTYoloV6::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes, float score_threshold,
                       float iou_threshold, unsigned int topk, unsigned int nms_type) {

    if (mat.empty()) return;
    // this->transform(mat);
    const int input_height = input_node_dims.at(2);
    const int input_width = input_node_dims.at(3);
    int img_height = static_cast<int>(mat.rows);
    int img_width = static_cast<int>(mat.cols);

    // resize & unscale
    cv::Mat mat_rs;
    YOLOv6ScaleParams scale_params;
    resize_unscale(mat, mat_rs, input_height, input_width, scale_params);

    normalized(mat_rs);

    // 1.make input
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

    // Synchronize the stream to ensure all operations are complete
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

