//
// Created by zhuohaoyang on 2026/8/27.
//

#include "trt_yolov26.h"
#include "lite/trt/core/trt_utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

using trtcv::TRTYoloV26;

namespace
{
  float clip(float value, float lower, float upper)
  {
    return std::min(std::max(value, lower), upper);
  }

  void check_cuda(cudaError_t status, const char *operation)
  {
    if (status != cudaSuccess)
      throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

TRTYoloV26::TRTYoloV26(const std::string &_trt_model_path,
                     unsigned int _num_threads) :
    BasicTRTHandler(_trt_model_path, _num_threads)
{
  if (!trt_engine || !trt_context || !stream)
    throw std::runtime_error("Failed to initialize YOLOV26 TensorRT engine");
  if (trt_engine->getNbIOTensors() != 2 || buffers.size() != 2)
    throw std::runtime_error("YOLOV26 expects exactly one input and one output tensor");

  const char *input_name = trt_engine->getIOTensorName(0);
  const char *output_name = trt_engine->getIOTensorName(1);
  if (trt_engine->getTensorIOMode(input_name) != nvinfer1::TensorIOMode::kINPUT ||
      trt_engine->getTensorIOMode(output_name) != nvinfer1::TensorIOMode::kOUTPUT)
    throw std::runtime_error("YOLOV26 expects the input tensor before the output tensor");
  if (trt_engine->getTensorDataType(input_name) != nvinfer1::DataType::kFLOAT ||
      trt_engine->getTensorDataType(output_name) != nvinfer1::DataType::kFLOAT)
    throw std::runtime_error("YOLOV26 expects float32 input and output tensors");

  if (input_node_dims.size() != 4 || input_node_dims[0] != 1 ||
      input_node_dims[1] != 3 || input_node_dims[2] <= 0 || input_node_dims[3] <= 0)
    throw std::runtime_error("YOLOV26 expects a static NCHW input with shape [1,3,H,W]");
  if (output_node_dims.size() != 1 || output_node_dims[0].size() != 3 ||
      output_node_dims[0][0] != 1 || output_node_dims[0][1] <= 0 ||
      output_node_dims[0][2] != 6)
    throw std::runtime_error("YOLOV26 expects one end-to-end output with shape [1,N,6]");
}

void TRTYoloV26::letterbox(const cv::Mat &mat, cv::Mat &mat_rs,
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

void TRTYoloV26::detect(const cv::Mat &mat,
                       std::vector<types::Boxf> &detected_boxes,
                       float score_threshold, unsigned int topk)
{
  detected_boxes.clear();
  if (mat.empty()) return;
  if (mat.channels() != 3)
    throw std::invalid_argument("YOLOV26 expects a three-channel BGR image");

  cv::Mat mat_rs;
  ScaleParams scale_params;
  this->letterbox(mat, mat_rs, static_cast<int>(input_node_dims[2]),
                  static_cast<int>(input_node_dims[3]), scale_params);
  cv::cvtColor(mat_rs, mat_rs, cv::COLOR_BGR2RGB);
  mat_rs.convertTo(mat_rs, CV_32FC3, 1.f / 255.f);

  std::vector<float> input;
  trtcv::utils::transform::create_tensor(
      mat_rs, input, input_node_dims, trtcv::utils::transform::CHW);

  const std::size_t input_size = static_cast<std::size_t>(input_node_dims[0]) *
                                 static_cast<std::size_t>(input_node_dims[1]) *
                                 static_cast<std::size_t>(input_node_dims[2]) *
                                 static_cast<std::size_t>(input_node_dims[3]);
  const auto &pred_dims = output_node_dims[0];
  const std::size_t num_predictions = static_cast<std::size_t>(pred_dims[1]);
  std::vector<float> output(num_predictions * 6);
  check_cuda(cudaMemcpyAsync(buffers[0], input.data(), input_size * sizeof(float),
                             cudaMemcpyHostToDevice, stream),
             "YOLOV26 input copy failed");
  if (!trt_context->enqueueV3(stream))
    throw std::runtime_error("Failed to infer YOLOV26 with TensorRT");
  check_cuda(cudaMemcpyAsync(output.data(), buffers[1], output.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, stream),
             "YOLOV26 output copy failed");
  check_cuda(cudaStreamSynchronize(stream), "YOLOV26 inference synchronization failed");

  this->generate_bboxes(scale_params, detected_boxes, output.data(),
                        num_predictions, score_threshold, topk,
                        mat.rows, mat.cols);
}

void TRTYoloV26::generate_bboxes(const ScaleParams &scale_params,
                                std::vector<types::Boxf> &detected_boxes,
                                const float *output,
                                std::size_t num_predictions,
                                float score_threshold, unsigned int topk,
                                int img_height, int img_width)
{
  detected_boxes.reserve(num_predictions);
  for (std::size_t i = 0; i < num_predictions; ++i)
  {
    const float *row = output + i * 6;
    const float score = row[4];
    const float class_value = row[5];
    if (!std::isfinite(score) || score < score_threshold ||
        !std::isfinite(class_value) || class_value < 0.f || class_value > 79.f)
      continue;

    const int label = static_cast<int>(std::round(class_value));
    if (std::fabs(class_value - static_cast<float>(label)) > 1e-3f)
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
