//
// Created by lite.ai.toolkit on 2026/8/26.
//

#include "yolo26.h"
#include "lite/ort/core/ort_utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

using ortcv::YOLO26;

namespace
{
  const char *const coco_class_names[80] = {
      "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
      "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
      "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
      "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
      "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
      "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
      "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
      "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
      "scissors", "teddy bear", "hair drier", "toothbrush"
  };

  float clip(float value, float lower, float upper)
  {
    return std::min(std::max(value, lower), upper);
  }
}

YOLO26::YOLO26(const std::string &_onnx_path, unsigned int _num_threads) :
    BasicOrtHandler(_onnx_path, _num_threads)
{
  if (input_node_dims.size() != 4 || input_node_dims[0] != 1 ||
      input_node_dims[1] != 3 || input_node_dims[2] <= 0 || input_node_dims[3] <= 0)
    throw std::runtime_error("YOLO26 expects a static NCHW input with shape [1,3,H,W]");

  if (num_outputs != 1 || output_node_dims[0].size() != 3 ||
      output_node_dims[0][0] != 1 || output_node_dims[0][2] != 6)
    throw std::runtime_error("YOLO26 expects one end-to-end output with shape [1,N,6]");

  Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(0);
  Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(0);
  auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
  auto output_info = output_type_info.GetTensorTypeAndShapeInfo();
  if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    throw std::runtime_error("YOLO26 expects float32 input and output tensors");
}

Ort::Value YOLO26::transform(const cv::Mat &mat_rs)
{
  cv::Mat canvas;
  cv::cvtColor(mat_rs, canvas, cv::COLOR_BGR2RGB);
  canvas.convertTo(canvas, CV_32FC3, 1.f / 255.f);
  return ortcv::utils::transform::create_tensor(
      canvas, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void YOLO26::letterbox(const cv::Mat &mat, cv::Mat &mat_rs,
                       int target_height, int target_width,
                       ScaleParams &scale_params)
{
  const float scale = std::min(
      static_cast<float>(target_width) / static_cast<float>(mat.cols),
      static_cast<float>(target_height) / static_cast<float>(mat.rows));
  const int resized_width = static_cast<int>(std::round(static_cast<float>(mat.cols) * scale));
  const int resized_height = static_cast<int>(std::round(static_cast<float>(mat.rows) * scale));
  const float half_pad_width = static_cast<float>(target_width - resized_width) / 2.f;
  const float half_pad_height = static_cast<float>(target_height - resized_height) / 2.f;
  const int left = static_cast<int>(std::round(half_pad_width - 0.1f));
  const int right = static_cast<int>(std::round(half_pad_width + 0.1f));
  const int top = static_cast<int>(std::round(half_pad_height - 0.1f));
  const int bottom = static_cast<int>(std::round(half_pad_height + 0.1f));

  cv::Mat resized;
  cv::resize(mat, resized, cv::Size(resized_width, resized_height));
  cv::copyMakeBorder(resized, mat_rs, top, bottom, left, right,
                     cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

  scale_params.scale = scale;
  scale_params.left = left;
  scale_params.top = top;
}

void YOLO26::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                    float score_threshold, unsigned int topk)
{
  detected_boxes.clear();
  if (mat.empty()) return;
  if (mat.channels() != 3)
    throw std::invalid_argument("YOLO26 expects a three-channel BGR image");

  cv::Mat mat_rs;
  ScaleParams scale_params;
  this->letterbox(mat, mat_rs, static_cast<int>(input_node_dims[2]),
                  static_cast<int>(input_node_dims[3]), scale_params);

  Ort::Value input_tensor = this->transform(mat_rs);
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1,
      output_node_names.data(), 1);

  this->generate_bboxes(scale_params, detected_boxes, output_tensors,
                        score_threshold, topk, mat.rows, mat.cols);
}

void YOLO26::generate_bboxes(const ScaleParams &scale_params,
                             std::vector<types::Boxf> &detected_boxes,
                             std::vector<Ort::Value> &output_tensors,
                             float score_threshold, unsigned int topk,
                             int img_height, int img_width)
{
  Ort::Value &pred = output_tensors.at(0);
  auto tensor_info = pred.GetTensorTypeAndShapeInfo();
  auto dims = tensor_info.GetShape();
  if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      dims.size() != 3 || dims[0] != 1 || dims[2] != 6 || dims[1] < 0)
    throw std::runtime_error("Unexpected YOLO26 output; expected float32 [1,N,6]");

  const float *data = pred.GetTensorData<float>();
  detected_boxes.reserve(static_cast<std::size_t>(dims[1]));
  for (int64_t i = 0; i < dims[1]; ++i)
  {
    const float *row = data + i * 6;
    const float score = row[4];
    const float class_value = row[5];
    if (!std::isfinite(score) || score < score_threshold ||
        !std::isfinite(class_value))
      continue;

    const int label = static_cast<int>(std::round(class_value));
    if (std::fabs(class_value - static_cast<float>(label)) > 1e-3f ||
        label < 0 || label >= 80)
      continue;

    float x1 = (row[0] - static_cast<float>(scale_params.left)) / scale_params.scale;
    float y1 = (row[1] - static_cast<float>(scale_params.top)) / scale_params.scale;
    float x2 = (row[2] - static_cast<float>(scale_params.left)) / scale_params.scale;
    float y2 = (row[3] - static_cast<float>(scale_params.top)) / scale_params.scale;
    if (!std::isfinite(x1) || !std::isfinite(y1) ||
        !std::isfinite(x2) || !std::isfinite(y2))
      continue;

    x1 = clip(x1, 0.f, static_cast<float>(img_width - 1));
    y1 = clip(y1, 0.f, static_cast<float>(img_height - 1));
    x2 = clip(x2, 0.f, static_cast<float>(img_width - 1));
    y2 = clip(y2, 0.f, static_cast<float>(img_height - 1));
    if (x2 <= x1 || y2 <= y1) continue;

    types::Boxf box;
    box.x1 = x1;
    box.y1 = y1;
    box.x2 = x2;
    box.y2 = y2;
    box.score = score;
    box.label = static_cast<unsigned int>(label);
    box.label_text = coco_class_names[label];
    box.flag = true;
    detected_boxes.push_back(box);
  }

  std::sort(detected_boxes.begin(), detected_boxes.end(),
            [](const types::Boxf &a, const types::Boxf &b)
            {
              return a.score > b.score;
            });
  if (detected_boxes.size() > static_cast<std::size_t>(topk))
    detected_boxes.resize(topk);
}
