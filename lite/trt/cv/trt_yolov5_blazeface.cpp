//
// Created by wangzijian on 7/27/24.
//

#include "trt_yolov5_blazeface.h"
using trtcv::TRTYOLO5Face;

void TRTYOLO5Face::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs, int target_height, int target_width,
                                  trtcv::TRTYOLO5Face::YOLOv5BlazeFaceScaleParams &scale_params) {

    if (mat.empty()) return;
    int img_height = static_cast<int>(mat.rows);
    int img_width = static_cast<int>(mat.cols);

    mat_rs = cv::Mat(target_height, target_width, CV_8UC3,
                     cv::Scalar(0, 0, 0));
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
    scale_params.ratio = r;
    scale_params.dw = dw;
    scale_params.dh = dh;
    scale_params.flag = true;

}

void TRTYOLO5Face::nms_bboxes_kps(std::vector<types::BoxfWithLandmarks> &input,
                                  std::vector<types::BoxfWithLandmarks> &output, float iou_threshold,
                                  unsigned int topk) {
    if (input.empty()) return;
    std::sort(
            input.begin(), input.end(),
            [](const types::BoxfWithLandmarks &a, const types::BoxfWithLandmarks &b)
            { return a.box.score > b.box.score; }
    );
    const unsigned int box_num = input.size();
    std::vector<int> merged(box_num, 0);

    unsigned int count = 0;
    for (unsigned int i = 0; i < box_num; ++i)
    {
        if (merged[i]) continue;
        std::vector<types::BoxfWithLandmarks> buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        for (unsigned int j = i + 1; j < box_num; ++j)
        {
            if (merged[j]) continue;

            float iou = static_cast<float>(input[i].box.iou_of(input[j].box));

            if (iou > iou_threshold)
            {
                merged[j] = 1;
                buf.push_back(input[j]);
            }

        }
        output.push_back(buf[0]);

        // keep top k
        count += 1;
        if (count >= topk)
            break;
    }
}

void TRTYOLO5Face::generate_bboxes_kps(const trtcv::TRTYOLO5Face::YOLOv5BlazeFaceScaleParams &scale_params,
                                       std::vector<types::BoxfWithLandmarks> &bbox_kps_collection, float *trt_outputs,
                                       float score_threshold, float img_height, float img_width) {

    int num_anchors = output_node_dims[0][1];

    float r_ = scale_params.ratio;
    int dw_ = scale_params.dw;
    int dh_ = scale_params.dh;

    bbox_kps_collection.clear();
    unsigned int count = 0;
    for (unsigned int i = 0; i < num_anchors; ++i)
    {
        const float *row_ptr = trt_outputs + i * 16;
        float obj_conf = row_ptr[4];
        if (obj_conf < score_threshold) continue; // filter first.
        float cls_conf = row_ptr[15];
        if (cls_conf < score_threshold) continue; // face score.

        // bounding box
        const float *offsets = row_ptr;
        float cx = offsets[0];
        float cy = offsets[1];
        float w = offsets[2];
        float h = offsets[3];

        types::BoxfWithLandmarks box_kps;
        float x1 = ((cx - w / 2.f) - (float) dw_) / r_;
        float y1 = ((cy - h / 2.f) - (float) dh_) / r_;
        float x2 = ((cx + w / 2.f) - (float) dw_) / r_;
        float y2 = ((cy + h / 2.f) - (float) dh_) / r_;
        box_kps.box.x1 = std::max(0.f, x1);
        box_kps.box.y1 = std::max(0.f, y1);
        box_kps.box.x2 = std::min(img_width - 1.f, x2);
        box_kps.box.y2 = std::min(img_height - 1.f, y2);
        box_kps.box.score = cls_conf;
        box_kps.box.label = 1;
        box_kps.box.label_text = "face";
        box_kps.box.flag = true;

        // landmarks
        const float *kps_offsets = row_ptr + 5;
        for (unsigned int j = 0; j < 10; j += 2)
        {
            cv::Point2f kps;
            float kps_x = (kps_offsets[j] - (float) dw_) / r_;
            float kps_y = (kps_offsets[j + 1] - (float) dh_) / r_;
            kps.x = std::min(std::max(0.f, kps_x), img_width - 1.f);
            kps.y = std::min(std::max(0.f, kps_y), img_height - 1.f);
            box_kps.landmarks.points.push_back(kps);
        }
        box_kps.landmarks.flag = true;
        box_kps.flag = true;

        bbox_kps_collection.push_back(box_kps);

        count += 1; // limit boxes for nms.
        if (count > max_nms)
            break;
    }

#if LITETRT_DEBUG
    std::cout << "generate_bboxes_kps num: " << bbox_kps_collection.size() << "\n";
#endif

}


void TRTYOLO5Face::normalized(cv::Mat &input_image) {
    cv::cvtColor(input_image,input_image,cv::COLOR_BGR2RGB);
    input_image.convertTo(input_image,CV_32F,scale_val,mean_val);
}


void TRTYOLO5Face::detect(const cv::Mat &mat, std::vector<types::BoxfWithLandmarks> &detected_boxes_kps,
                          float score_threshold, float iou_threshold, unsigned int topk) {
    if (mat.empty()) return;
    auto img_height = static_cast<float>(mat.rows);
    auto img_width = static_cast<float>(mat.cols);
    const int target_height = (int) input_node_dims.at(2);
    const int target_width = (int) input_node_dims.at(3);
    // resize & unscale
    cv::Mat mat_rs;
    YOLOv5BlazeFaceScaleParams scale_params;
    resize_unscale(mat, mat_rs, target_height, target_width, scale_params);

    normalized(mat_rs);

    std::vector<float> input;
    trtcv::utils::transform::create_tensor(mat_rs,input,input_node_dims,trtcv::utils::transform::CHW);

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
    generate_bboxes_kps(scale_params, detected_boxes_kps, output.data(),
                              score_threshold, img_height, img_width);

    nms_bboxes_kps(detected_boxes_kps, detected_boxes_kps, iou_threshold, topk);

}