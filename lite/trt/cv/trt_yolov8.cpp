//
// Created by wangzijian on 7/24/24.
//

#include "trt_yolov8.h"
using trtcv::TRTYoloV8;


void TRTYoloV8::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                    float iou_threshold, unsigned int topk, unsigned int nms_type)
{
    if (nms_type == NMS::BLEND) lite::utils::blending_nms(input, output, iou_threshold, topk);
    else if (nms_type == NMS::OFFSET) lite::utils::offset_nms(input, output, iou_threshold, topk);
    else lite::utils::hard_nms(input, output, iou_threshold, topk);
}

void TRTYoloV8::generate_bboxes(std::vector<types::Boxf> &bbox_collection, float* output, float score_threshold,
                                int img_height, int img_width) {
    auto pred_dims = output_node_dims[0];
    const unsigned int num_anchors = pred_dims[2]; // 8400
    const unsigned int num_classes = pred_dims[1] - 4; // 80

    float x_factor = float(img_width) / input_node_dims[3];
    float y_factor = float(img_height) / input_node_dims[2];

    bbox_collection.clear();
    unsigned int count = 0;

    for (unsigned int i = 0; i < num_anchors; ++i) {

        std::vector<float> class_scores(num_classes);
        for (unsigned int j = 0; j < num_classes; ++j) {
            class_scores[j] = output[(4 + j) * num_anchors + i];
        }

        auto max_it = std::max_element(class_scores.begin(), class_scores.end());
        float max_cls_conf = *max_it;
        unsigned int label = std::distance(class_scores.begin(), max_it);

        float conf = max_cls_conf;
        if (conf < score_threshold) continue;

        float cx = output[0 * num_anchors + i];
        float cy = output[1 * num_anchors + i];
        float w = output[2 * num_anchors + i];
        float h = output[3 * num_anchors + i];

        float x1 = (cx - w / 2.f)  * x_factor;
        float y1 = (cy - h / 2.f)  * y_factor;

        w = w * x_factor;
        h = h * y_factor;

        float x2 = x1 + w ;
        float y2 = y1 + h;

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

        count += 1;
        if (count > max_nms)
            break;
    }

#if LITETRT_DEBUG
    std::cout << "detected num_anchors: " << num_anchors << "\n";
    std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
#endif
}

void TRTYoloV8::preprocess(cv::Mat &input_image) {

    // Convert color space from BGR to RGB
    cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);

    // Resize image
    cv::resize(input_image, input_image, cv::Size(input_node_dims[2], input_node_dims[3]), 0, 0, cv::INTER_LINEAR);

    // Normalize image
    input_image.convertTo(input_image, CV_32F, scale_val, mean_val);
}


void TRTYoloV8::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes, float score_threshold,
                       float iou_threshold, unsigned int topk, unsigned int nms_type) {

    if (mat.empty()) return;
    int img_height = static_cast<int>(mat.rows);
    int img_width = static_cast<int>(mat.cols);

    // resize & unscale
    cv::Mat mat_rs = mat.clone();

    preprocess(mat_rs);

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

    std::vector<types::Boxf> bbox_collection;
    generate_bboxes(bbox_collection,output.data(),score_threshold,img_height,img_width);
    nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);

}

