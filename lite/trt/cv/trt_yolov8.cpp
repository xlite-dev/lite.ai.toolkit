//
// Created by wangzijian on 7/24/24.
//

#include "trt_yolov8.h"
using trtcv::TRTYoloV8;


void TRTYoloV8::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                               int target_height, int target_width,
                               YoloV8ScaleParams &scale_params)
{
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


void TRTYoloV8::normalized(const cv::Mat &input_image) {

    cv::cvtColor(input_image,input_image,cv::COLOR_BGR2RGB);
    input_image.convertTo(input_image,CV_32F,1.0 / 255.0,0);
}


void TRTYoloV8::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                    float iou_threshold, unsigned int topk, unsigned int nms_type)
{
    if (nms_type == NMS::BLEND) lite::utils::blending_nms(input, output, iou_threshold, topk);
    else if (nms_type == NMS::OFFSET) lite::utils::offset_nms(input, output, iou_threshold, topk);
    else lite::utils::hard_nms(input, output, iou_threshold, topk);
}

void TRTYoloV8::generate_bboxes(const trtcv::TRTYoloV8::YoloV8ScaleParams &scale_params,
                                std::vector<types::Boxf> &bbox_collection, float *output, float score_threshold,
                                int img_height, int img_width) {
    auto pred_dims = output_node_dims[0]; // 获取输出的维度 这里是 1 * 84 * 8400 不能直接炒上面yolox和yolov5的代码
    const unsigned int num_classes = pred_dims.at(1) - 4; // n = ?
    const unsigned int num_anchors = pred_dims.at(2);


    float r_ = scale_params.r;
    int dw_ = scale_params.dw;
    int dh_ = scale_params.dh;

    bbox_collection.clear();

    unsigned int count = 0;

    // yolov8 修改了 他的输出维度 他把类别置信度的最大值和分类的值混在一起了

    for (unsigned int i = 0; i < num_anchors; ++i)
    {
        float tmp_conf_score= output[i * num_classes + 4 + i]; //得到xywh之后的第一个类别置信度

    }



}


void TRTYoloV8::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes, float score_threshold,
                       float iou_threshold, unsigned int topk, unsigned int nms_type) {











}

