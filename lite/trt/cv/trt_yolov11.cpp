//
// Created by wangzijian.
//

#include "trt_yolov11.h"
using trtcv::TRTYOLOV11;

// NMS 实现
void TRTYOLOV11::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                    float iou_threshold, unsigned int topk, unsigned int nms_type)
{
    if (nms_type == NMS::BLEND) lite::utils::blending_nms(input, output, iou_threshold, topk);
    else if (nms_type == NMS::OFFSET) lite::utils::offset_nms(input, output, iou_threshold, topk);
    else lite::utils::hard_nms(input, output, iou_threshold, topk);
}

// 解析 Bounding Boxes
// 修改：不再依赖 img_height/width 计算因子，而是使用预处理时计算出的 scale 和 padding
void TRTYOLOV11::generate_bboxes(std::vector<types::Boxf> &bbox_collection, float* output, float score_threshold,
                                float scale, float pad_w, float pad_h) {
    auto pred_dims = output_node_dims[0]; // [1, 84, 8400]
    const unsigned int num_anchors = pred_dims[2];
    const unsigned int num_classes = pred_dims[1] - 4;

    bbox_collection.clear();
    unsigned int count = 0;

    for (unsigned int i = 0; i < num_anchors; ++i) {
        float max_cls_conf = -1.f;
        unsigned int label = 0;

        // 寻找最大类别分数
        for (unsigned int j = 0; j < num_classes; ++j) {
            float cls_score = output[(4 + j) * num_anchors + i];
            if (cls_score > max_cls_conf) {
                max_cls_conf = cls_score;
                label = j;
            }
        }

        if (max_cls_conf < score_threshold) continue;

        // 解析坐标 (模型输出的是 LetterBox 后的坐标)
        float cx = output[0 * num_anchors + i];
        float cy = output[1 * num_anchors + i];
        float w = output[2 * num_anchors + i];
        float h = output[3 * num_anchors + i];

        // 1. 还原到左上角坐标 (x1, y1)
        float x1_net = cx - w / 2.f;
        float y1_net = cy - h / 2.f;

        // 2. 移除 LetterBox 的 Padding 并缩放回原图
        float x1 = (x1_net - pad_w) / scale;
        float y1 = (y1_net - pad_h) / scale;
        float w_original = w / scale;
        float h_original = h / scale;

        float x2 = x1 + w_original;
        float y2 = y1 + h_original;

        types::Boxf box;
        // 这里的边界保护通常不需要，或者可以在外层做，这里保留
        box.x1 = std::max(0.f, x1);
        box.y1 = std::max(0.f, y1);
        box.x2 = x2;
        box.y2 = y2;
        box.score = max_cls_conf;
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

// 新增：LetterBox 实现
void TRTYOLOV11::letterbox(const cv::Mat &image, cv::Mat &out_image,
                           const cv::Size &new_shape,
                           int stride, const cv::Scalar &color,
                           bool fixed_shape, bool scale_up) {
    cv::Size shape = image.size();
    float r = std::min((float)new_shape.height / (float)shape.height,
                       (float)new_shape.width / (float)shape.width);
    if (!scale_up) {
        r = std::min(r, 1.0f);
    }

    int new_unpad_w = int(round(shape.width * r));
    int new_unpad_h = int(round(shape.height * r));
    int dw = new_shape.width - new_unpad_w;
    int dh = new_shape.height - new_unpad_h;

    // 如果是固定形状（通常推理时需要固定输入尺寸），直接除以2居中
    if (fixed_shape) {
        dw = dw % stride;
        dh = dh % stride;
    }

    dw /= 2;
    dh /= 2;

    if (shape.width != new_unpad_w || shape.height != new_unpad_h) {
        cv::resize(image, out_image, cv::Size(new_unpad_w, new_unpad_h));
    } else {
        out_image = image;
    }

    int top = int(round(dh - 0.1));
    int bottom = int(round(dh + 0.1));
    int left = int(round(dw - 0.1));
    int right = int(round(dw + 0.1));

    // 填充边框
    cv::copyMakeBorder(out_image, out_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);

    // 强制 resize 到确切的目标尺寸 (处理 odd padding 的情况)
    if (out_image.size() != new_shape) {
        cv::resize(out_image, out_image, new_shape);
    }
}

// 预处理 (仅负责颜色转换和归一化，LetterBox 移到 detect 中计算参数)
void TRTYOLOV11::preprocess(cv::Mat &input_image) {
    // 1. Convert BGR -> RGB
    cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);
    // 2. Normalize (0-255 -> 0.0-1.0)
    input_image.convertTo(input_image, CV_32F, scale_val, mean_val);
}

// 推理主函数
void TRTYOLOV11::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes, float score_threshold,
                       float iou_threshold, unsigned int topk, unsigned int nms_type) {

    if (mat.empty()) return;

    // --- LetterBox Preprocessing ---
    // 1. 计算缩放比例和 Padding，用于后续坐标还原
    int target_h = input_node_dims[2];
    int target_w = input_node_dims[3];
    int img_h = mat.rows;
    int img_w = mat.cols;

    // 计算缩放比 r
    float r = std::min((float)target_h / img_h, (float)target_w / img_w);
    int new_unpad_w = int(round(img_w * r));
    int new_unpad_h = int(round(img_h * r));

    // 计算 padding (dw, dh)
    // 注意：TensorRT 推理通常需要严格的输入尺寸，所以 pad 到 target_w/h
    int dw = (target_w - new_unpad_w) / 2;
    int dh = (target_h - new_unpad_h) / 2;

    // 执行 Resize + Pad
    cv::Mat mat_rs;
    if (img_h != new_unpad_h || img_w != new_unpad_w) {
        cv::resize(mat, mat_rs, cv::Size(new_unpad_w, new_unpad_h));
    } else {
        mat_rs = mat.clone();
    }

    // 填充边界到目标尺寸
    int top = dh;
    int bottom = target_h - new_unpad_h - top;
    int left = dw;
    int right = target_w - new_unpad_w - left;

    cv::copyMakeBorder(mat_rs, mat_rs, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    // -------------------------------

    preprocess(mat_rs);

    // 1. Make the input (HWC -> CHW)
    std::vector<float> input;
    trtcv::utils::transform::create_tensor(mat_rs, input, input_node_dims, trtcv::utils::transform::CHW);

    // 2. Inference
    cudaMemcpyAsync(buffers[0], input.data(),
                    input_node_dims[0] * input_node_dims[1] * input_node_dims[2] * input_node_dims[3] * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    cudaStreamSynchronize(stream);

    bool status = trt_context->enqueueV3(stream); // TensorRT 8.5+ usage
    if (!status){
         std::cerr << "Failed to infer by TensorRT." << std::endl;
         return;
    }

    cudaStreamSynchronize(stream);

    // D -> H
    auto pred_dims = output_node_dims[0];
    size_t output_size = pred_dims[0] * pred_dims[1] * pred_dims[2];
    std::vector<float> output(output_size);

    cudaMemcpyAsync(output.data(), buffers[1], output_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 3. Post-process (传入 r 作为 scale, dw/dh 作为 padding)
    std::vector<types::Boxf> bbox_collection;

    // 注意: letterbox 会导致 0.5 像素的坐标偏移，这里为了简化直接传 float padding
    // 实际上更严谨的做法是在 generate_bboxes 里用 (x - left) / r
    generate_bboxes(bbox_collection, output.data(), score_threshold, r, (float)left, (float)top);

    nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}