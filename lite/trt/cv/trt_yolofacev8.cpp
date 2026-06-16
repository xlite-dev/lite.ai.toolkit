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



std::vector<int>
TRTYoloFaceV8::nms_cuda(std::vector<lite::types::Boxf> boxes, std::vector<float> confidences, const float nms_thresh) {
        return nms_cuda_manager->perform_nms(boxes, confidences, nms_thresh);
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

void TRTYoloFaceV8::generate_box(float *trt_outputs, std::vector<lite::types::Boxf> &boxes, float conf_threshold,
                                 float iou_threshold) {

    int num_box = output_node_dims[0][2];

    // 直接分配目标类型的向量
    std::vector<lite::types::BoundingBoxType<float, float>> bounding_box_raw(num_box);

    // 调用包装函数
    launch_yolov8_postprocess(
            static_cast<float*>(buffers[1]),
            num_box,
            conf_threshold,
            ratio_height,
            ratio_width,
            bounding_box_raw.data(),
            num_box
    );

    std::vector<float> score_raw;
    for (const auto& bbox : bounding_box_raw) {
        if (bbox.score >= 0) {
            score_raw.emplace_back(bbox.score);
        }
    }



    std::vector<int> keep_inds = nms_cuda(bounding_box_raw, score_raw, iou_threshold);
//    std::vector<int> keep_inds = this->nms(bounding_box_raw, score_raw, iou_threshold);

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

    // 检查输入
    if (mat.empty()) {
        std::cerr << "Input image is empty!" << std::endl;
        return;
    }

    // 检查 TRT 上下文
    if (!trt_context) {
        std::cerr << "TensorRT context is null!" << std::endl;
        return;
    }


    // 1. letterbox: resize (keep aspect) + pad to the network input size, BGR uint8. Sets ratio_*.
    const int input_height = input_node_dims[2];
    const int input_width  = input_node_dims[3];
    cv::Mat temp_image = mat;
    if (mat.rows > input_height || mat.cols > input_width) {
        const float s = std::min((float)input_height / mat.rows, (float)input_width / mat.cols);
        cv::resize(mat, temp_image, cv::Size(int(mat.cols * s), int(mat.rows * s)));
    }
    ratio_height = (float)mat.rows / temp_image.rows;
    ratio_width  = (float)mat.cols / temp_image.cols;
    cv::Mat input_img;
    // BORDER_ISOLATED: when `mat` is a ROI/submatrix of a larger image and no resize
    // happened (temp_image == mat), plain copyMakeBorder would pull the parent image's
    // pixels (outside the ROI) into the pad region instead of the constant. Isolating the
    // ROI restores the old clone()-based behavior.
    cv::copyMakeBorder(temp_image, input_img, 0, input_height - temp_image.rows,
                       0, input_width - temp_image.cols,
                       cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, 0);

    // 2. GPU-fused normalize + BGR HWC->CHW straight into the inference input buffer
    //    (replaces CPU split / 3x convertTo / merge / create_tensor + the separate float H2D).
    preprocess_gpu_.run(input_img, static_cast<float*>(buffers[0]), stream);

    // 3. infer
    bool status = trt_context->enqueueV3(stream);
    if (!status){
        std::cerr << "Failed to infer by TensorRT." << std::endl;
        return;
    }
    cudaStreamSynchronize(stream);   // ensure the inference output (buffers[1]) is ready

    // 4. generate box (reads buffers[1] directly; the trt_outputs param is unused)
    generate_box(nullptr, boxes, 0.45f, 0.5f);


}
