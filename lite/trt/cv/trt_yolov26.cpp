//
// Created by zhuohaoyang on 2026/8/27.
//

#include "trt_yolov26.h"
#include "lite/trt/core/trt_utils.h"
#include "lite/trt/kernel/yolov26_preprocess.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

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

  double elapsed_ms(const std::chrono::steady_clock::time_point &start,
                    const std::chrono::steady_clock::time_point &end)
  {
    return std::chrono::duration<double, std::milli>(end - start).count();
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

  const std::size_t input_size = static_cast<std::size_t>(input_node_dims[0]) *
                                 static_cast<std::size_t>(input_node_dims[1]) *
                                 static_cast<std::size_t>(input_node_dims[2]) *
                                 static_cast<std::size_t>(input_node_dims[3]);
  const std::size_t output_size = static_cast<std::size_t>(output_node_dims[0][1]) * 6;
  host_input.resize(input_size);
  host_output.resize(output_size);

  cudaError_t status = cudaHostRegister(host_input.data(),
                                        host_input.size() * sizeof(float),
                                        cudaHostRegisterDefault);
  if (status == cudaSuccess)
    host_input_registered = true;
  else
    cudaGetLastError();

  status = cudaHostRegister(host_output.data(),
                            host_output.size() * sizeof(float),
                            cudaHostRegisterDefault);
  if (status == cudaSuccess)
    host_output_registered = true;
  else
    cudaGetLastError();
}

TRTYoloV26::~TRTYoloV26()
{
  if (stream) cudaStreamSynchronize(stream);

  if (device_image)
  {
    cudaFree(device_image);
    device_image = nullptr;
  }
  if (host_image)
  {
    cudaFreeHost(host_image);
    host_image = nullptr;
  }
  image_capacity = 0;

  if (host_output_registered)
  {
    cudaHostUnregister(host_output.data());
    host_output_registered = false;
  }
  if (host_input_registered)
  {
    cudaHostUnregister(host_input.data());
    host_input_registered = false;
  }

  for (auto &event : timing_events)
  {
    if (event)
    {
      cudaEventDestroy(event);
      event = nullptr;
    }
  }
}

TRTYoloV26::ScaleParams TRTYoloV26::calculate_scale_params(const cv::Mat &mat) const
{
  const int target_height = static_cast<int>(input_node_dims[2]);
  const int target_width = static_cast<int>(input_node_dims[3]);
  ScaleParams params{};
  params.scale = std::min(
      static_cast<float>(target_width) / static_cast<float>(mat.cols),
      static_cast<float>(target_height) / static_cast<float>(mat.rows));
  params.resized_width = static_cast<int>(
      std::round(static_cast<float>(mat.cols) * params.scale));
  params.resized_height = static_cast<int>(
      std::round(static_cast<float>(mat.rows) * params.scale));
  if (params.resized_width <= 0 || params.resized_height <= 0)
    throw std::invalid_argument("YOLOV26 input aspect ratio produces an empty resize");
  params.left = (target_width - params.resized_width) / 2;
  params.top = (target_height - params.resized_height) / 2;
  return params;
}

void TRTYoloV26::preprocess_cpu(const cv::Mat &mat, std::vector<float> &input,
                                ScaleParams &scale_params)
{
  scale_params = calculate_scale_params(mat);
  const int target_height = static_cast<int>(input_node_dims[2]);
  const int target_width = static_cast<int>(input_node_dims[3]);
  const int right = target_width - scale_params.resized_width - scale_params.left;
  const int bottom = target_height - scale_params.resized_height - scale_params.top;

  cv::Mat resized;
  cv::Mat letterboxed;
  cv::resize(mat, resized,
             cv::Size(scale_params.resized_width, scale_params.resized_height));
  cv::copyMakeBorder(resized, letterboxed, scale_params.top, bottom,
                     scale_params.left, right, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));
  cv::cvtColor(letterboxed, letterboxed, cv::COLOR_BGR2RGB);
  letterboxed.convertTo(letterboxed, CV_32FC3, 1.f / 255.f);
  trtcv::utils::transform::create_tensor(
      letterboxed, input, input_node_dims, trtcv::utils::transform::CHW);
}

void TRTYoloV26::ensure_timing_events()
{
  if (timing_events[0]) return;
  for (auto &event : timing_events)
  {
    const cudaError_t status = cudaEventCreate(&event);
    if (status != cudaSuccess)
    {
      for (auto &created_event : timing_events)
      {
        if (created_event)
        {
          cudaEventDestroy(created_event);
          created_event = nullptr;
        }
      }
      check_cuda(status, "Failed to create YOLOV26 timing event");
    }
  }
}

bool TRTYoloV26::ensure_image_buffers(std::size_t required_bytes)
{
  if (required_bytes <= image_capacity) return true;
  check_cuda(cudaStreamSynchronize(stream),
             "Failed to synchronize before growing YOLOV26 image buffers");

  unsigned char *new_host = nullptr;
  unsigned char *new_device = nullptr;
  const cudaError_t host_status = cudaHostAlloc(
      reinterpret_cast<void **>(&new_host), required_bytes, cudaHostAllocDefault);
  if (host_status != cudaSuccess)
  {
    cudaGetLastError();
    return false;
  }
  const cudaError_t device_status = cudaMalloc(
      reinterpret_cast<void **>(&new_device), required_bytes);
  if (device_status != cudaSuccess)
  {
    cudaFreeHost(new_host);
    check_cuda(device_status, "Failed to allocate YOLOV26 device image buffer");
  }

  if (device_image) cudaFree(device_image);
  if (host_image) cudaFreeHost(host_image);
  host_image = new_host;
  device_image = new_device;
  image_capacity = required_bytes;
  return true;
}

bool TRTYoloV26::pack_image(const cv::Mat &mat)
{
  if (mat.type() != CV_8UC3)
    throw std::invalid_argument("YOLOV26 fused CUDA preprocessing requires CV_8UC3 input");
  const std::size_t row_bytes = static_cast<std::size_t>(mat.cols) * mat.elemSize();
  const std::size_t required_bytes = row_bytes * static_cast<std::size_t>(mat.rows);
  if (!ensure_image_buffers(required_bytes)) return false;

  if (mat.isContinuous())
  {
    std::memcpy(host_image, mat.data, required_bytes);
    return true;
  }
  for (int row = 0; row < mat.rows; ++row)
    std::memcpy(host_image + static_cast<std::size_t>(row) * row_bytes,
                mat.ptr(row), row_bytes);
  return true;
}

void TRTYoloV26::enqueue_cuda_preprocess(const cv::Mat &mat,
                                         const ScaleParams &scale_params)
{
  const std::size_t row_bytes = static_cast<std::size_t>(mat.cols) * mat.elemSize();
  check_cuda(trtcv::kernel::launch_yolov26_preprocess(
                 device_image, row_bytes, mat.cols, mat.rows,
                 static_cast<float *>(buffers[0]),
                 static_cast<int>(input_node_dims[3]),
                 static_cast<int>(input_node_dims[2]),
                 scale_params.resized_width, scale_params.resized_height,
                 scale_params.left, scale_params.top, stream),
             "Failed to launch YOLOV26 fused CUDA preprocessing");
}

void TRTYoloV26::detect(const cv::Mat &mat,
                        std::vector<types::Boxf> &detected_boxes,
                        float score_threshold, unsigned int topk)
{
  detect_impl(mat, detected_boxes, score_threshold, topk, nullptr,
              PipelineMode::Optimized);
}

void TRTYoloV26::detect(const cv::Mat &mat,
                        std::vector<types::Boxf> &detected_boxes,
                        float score_threshold, unsigned int topk,
                        PipelineMode mode)
{
  detect_impl(mat, detected_boxes, score_threshold, topk, nullptr, mode);
}

void TRTYoloV26::detect_with_timing(const cv::Mat &mat,
                                    std::vector<types::Boxf> &detected_boxes,
                                    Timing &timing,
                                    float score_threshold, unsigned int topk)
{
  detect_impl(mat, detected_boxes, score_threshold, topk, &timing,
              PipelineMode::Optimized);
}

void TRTYoloV26::detect_with_timing(const cv::Mat &mat,
                                    std::vector<types::Boxf> &detected_boxes,
                                    Timing &timing,
                                    float score_threshold, unsigned int topk,
                                    PipelineMode mode)
{
  detect_impl(mat, detected_boxes, score_threshold, topk, &timing, mode);
}

void TRTYoloV26::detect_impl(const cv::Mat &mat,
                             std::vector<types::Boxf> &detected_boxes,
                             float score_threshold, unsigned int topk,
                             Timing *timing, PipelineMode mode)
{
  using Clock = std::chrono::steady_clock;

  detected_boxes.clear();
  if (timing) *timing = Timing{};
  const auto total_start = Clock::now();

  if (mat.empty()) return;
  if (mat.channels() != 3)
    throw std::invalid_argument("YOLOV26 expects a three-channel BGR image");
  if (mode == PipelineMode::Optimized && mat.type() != CV_8UC3)
    mode = PipelineMode::PinnedCpu;
  if (timing) ensure_timing_events();

  std::vector<float> baseline_input;
  std::vector<float> baseline_output;
  std::vector<float> &input = mode == PipelineMode::Baseline ? baseline_input : host_input;
  ScaleParams scale_params{};

  const auto preprocess_start = Clock::now();
  if (mode == PipelineMode::Optimized)
  {
    scale_params = calculate_scale_params(mat);
    if (!pack_image(mat)) mode = PipelineMode::PinnedCpu;
  }
  if (mode != PipelineMode::Optimized)
  {
    const float *registered_input = mode == PipelineMode::PinnedCpu
                                        ? host_input.data() : nullptr;
    preprocess_cpu(mat, input, scale_params);
    if (mode == PipelineMode::PinnedCpu && input.data() != registered_input)
      throw std::runtime_error("YOLOV26 reusable input buffer was unexpectedly reallocated");
  }
  const auto preprocess_end = Clock::now();

  const auto backend_start = Clock::now();
  const auto &pred_dims = output_node_dims[0];
  const std::size_t num_predictions = static_cast<std::size_t>(pred_dims[1]);
  if (mode == PipelineMode::Baseline)
    baseline_output.resize(num_predictions * 6);
  std::vector<float> &output = mode == PipelineMode::Baseline
                                   ? baseline_output : host_output;

  bool stream_work_submitted = false;
  try
  {
    if (timing)
      check_cuda(cudaEventRecord(timing_events[0], stream),
                 "Failed to record YOLOV26 input H2D start");
    if (mode == PipelineMode::Optimized)
    {
      const std::size_t image_bytes = static_cast<std::size_t>(mat.rows) *
                                      static_cast<std::size_t>(mat.cols) * mat.elemSize();
      check_cuda(cudaMemcpyAsync(device_image, host_image, image_bytes,
                                 cudaMemcpyHostToDevice, stream),
                 "YOLOV26 raw image copy failed");
    }
    else
    {
      check_cuda(cudaMemcpyAsync(buffers[0], input.data(), input.size() * sizeof(float),
                                 cudaMemcpyHostToDevice, stream),
                 "YOLOV26 input tensor copy failed");
    }
    stream_work_submitted = true;
    if (timing)
      check_cuda(cudaEventRecord(timing_events[1], stream),
                 "Failed to record YOLOV26 input H2D end");

    if (mode == PipelineMode::Optimized)
      enqueue_cuda_preprocess(mat, scale_params);
    if (timing)
      check_cuda(cudaEventRecord(timing_events[2], stream),
                 "Failed to record YOLOV26 inference start");

    if (!trt_context->enqueueV3(stream))
      throw std::runtime_error("Failed to infer YOLOV26 with TensorRT");
    if (timing)
      check_cuda(cudaEventRecord(timing_events[3], stream),
                 "Failed to record YOLOV26 inference end");

    check_cuda(cudaMemcpyAsync(output.data(), buffers[1], output.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream),
               "YOLOV26 output copy failed");
    if (timing)
      check_cuda(cudaEventRecord(timing_events[4], stream),
                 "Failed to record YOLOV26 output D2H end");
    check_cuda(cudaStreamSynchronize(stream),
               "YOLOV26 inference synchronization failed");
    stream_work_submitted = false;
  }
  catch (...)
  {
    if (stream_work_submitted) cudaStreamSynchronize(stream);
    throw;
  }
  const auto backend_end = Clock::now();

  if (timing)
  {
    float duration = 0.f;
    check_cuda(cudaEventElapsedTime(&duration, timing_events[0], timing_events[1]),
               "Failed to measure YOLOV26 H2D time");
    timing->h2d_ms = duration;
    if (mode == PipelineMode::Optimized)
    {
      check_cuda(cudaEventElapsedTime(&duration, timing_events[1], timing_events[2]),
                 "Failed to measure YOLOV26 GPU preprocess time");
      timing->gpu_preprocess_ms = duration;
    }
    check_cuda(cudaEventElapsedTime(&duration, timing_events[2], timing_events[3]),
               "Failed to measure YOLOV26 inference time");
    timing->inference_ms = duration;
    check_cuda(cudaEventElapsedTime(&duration, timing_events[3], timing_events[4]),
               "Failed to measure YOLOV26 D2H time");
    timing->d2h_ms = duration;
  }

  const auto postprocess_start = Clock::now();
  generate_bboxes(scale_params, detected_boxes, output.data(),
                  num_predictions, score_threshold, topk,
                  mat.rows, mat.cols);
  const auto postprocess_end = Clock::now();

  if (timing)
  {
    timing->preprocess_ms = elapsed_ms(preprocess_start, preprocess_end);
    timing->backend_wall_ms = elapsed_ms(backend_start, backend_end);
    timing->postprocess_ms = elapsed_ms(postprocess_start, postprocess_end);
    timing->total_ms = elapsed_ms(total_start, postprocess_end);
  }
}

TRTYoloV26::PreprocessComparison TRTYoloV26::compare_preprocess(const cv::Mat &mat)
{
  if (mat.empty() || mat.type() != CV_8UC3)
    throw std::invalid_argument("YOLOV26 preprocess comparison requires non-empty CV_8UC3 input");

  ScaleParams cpu_params{};
  std::vector<float> cpu_input;
  preprocess_cpu(mat, cpu_input, cpu_params);
  if (!pack_image(mat))
    throw std::runtime_error("Failed to allocate YOLOV26 validation staging buffer");

  const std::size_t image_bytes = static_cast<std::size_t>(mat.rows) *
                                  static_cast<std::size_t>(mat.cols) * mat.elemSize();
  std::vector<float> gpu_input(cpu_input.size());
  bool stream_work_submitted = false;
  try
  {
    check_cuda(cudaMemcpyAsync(device_image, host_image, image_bytes,
                               cudaMemcpyHostToDevice, stream),
               "YOLOV26 validation image copy failed");
    stream_work_submitted = true;
    enqueue_cuda_preprocess(mat, cpu_params);
    check_cuda(cudaMemcpyAsync(gpu_input.data(), buffers[0],
                               gpu_input.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream),
               "YOLOV26 validation tensor copy failed");
    check_cuda(cudaStreamSynchronize(stream),
               "YOLOV26 preprocess validation synchronization failed");
    stream_work_submitted = false;
  }
  catch (...)
  {
    if (stream_work_submitted) cudaStreamSynchronize(stream);
    throw;
  }

  PreprocessComparison comparison;
  comparison.elements = cpu_input.size();
  std::vector<float> errors;
  errors.reserve(cpu_input.size());
  double error_sum = 0.0;
  for (std::size_t i = 0; i < cpu_input.size(); ++i)
  {
    const float error = std::fabs(cpu_input[i] - gpu_input[i]);
    if (!std::isfinite(error))
    {
      ++comparison.mismatched;
      errors.push_back(std::numeric_limits<float>::infinity());
      comparison.max_abs_error = std::numeric_limits<float>::infinity();
      error_sum = std::numeric_limits<double>::infinity();
      continue;
    }
    errors.push_back(error);
    error_sum += static_cast<double>(error);
    if (error != 0.0f) ++comparison.mismatched;
    comparison.max_abs_error = std::max(comparison.max_abs_error, error);
  }
  comparison.mean_abs_error = error_sum / static_cast<double>(comparison.elements);
  std::sort(errors.begin(), errors.end());
  const std::size_t p99_index = static_cast<std::size_t>(
      std::ceil(0.99 * static_cast<double>(errors.size()))) - 1;
  comparison.p99_abs_error = errors[std::min(p99_index, errors.size() - 1)];
  return comparison;
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
