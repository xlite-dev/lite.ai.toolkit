#include "lite/lite.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/yolov26n-640x640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_detection_1.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolov26_1.jpg";

  // 1. Test Default Engine ONNXRuntime
  lite::cv::detection::YoloV26 *yolov26 =
      new lite::cv::detection::YoloV26(onnx_path); // default

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov26->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete yolov26;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../examples/hub/onnx/cv/yolov26n-640x640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_detection_1.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolov26_2.jpg";

  // 2. Test Specific Engine ONNXRuntime
  lite::onnxruntime::cv::detection::YoloV26 *yolov26 =
      new lite::onnxruntime::cv::detection::YoloV26(onnx_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov26->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete yolov26;
#endif
}

static void test_tensorrt()
{
#ifdef ENABLE_TENSORRT
  std::string engine_path = "../../../examples/hub/trt/yolov26n_fp32.engine";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_detection_1.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolov26_1_trt.jpg";

  // 3. Test Specific Engine TensorRT
  lite::trt::cv::detection::YOLOV26 *yolov26 =
      new lite::trt::cv::detection::YOLOV26(engine_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov26->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "TensorRT Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete yolov26;
#endif
}

static void test_lite()
{
  test_default();
  test_onnxruntime();
  test_tensorrt();
}

#ifdef ENABLE_TENSORRT
namespace
{
  using Detector = lite::trt::cv::detection::YOLOV26;
  using Timing = Detector::Timing;
  using PipelineMode = Detector::PipelineMode;

  struct Statistics
  {
    double mean = 0.0;
    double minimum = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
  };

  struct Samples
  {
    std::vector<double> preprocess;
    std::vector<double> h2d;
    std::vector<double> gpu_preprocess;
    std::vector<double> inference;
    std::vector<double> d2h;
    std::vector<double> gpu_pipeline;
    std::vector<double> backend_wall;
    std::vector<double> postprocess;
    std::vector<double> total;

    void reserve(std::size_t count)
    {
      preprocess.reserve(count);
      h2d.reserve(count);
      gpu_preprocess.reserve(count);
      inference.reserve(count);
      d2h.reserve(count);
      gpu_pipeline.reserve(count);
      backend_wall.reserve(count);
      postprocess.reserve(count);
      total.reserve(count);
    }

    void append(const Timing &timing)
    {
      preprocess.push_back(timing.preprocess_ms);
      h2d.push_back(timing.h2d_ms);
      gpu_preprocess.push_back(timing.gpu_preprocess_ms);
      inference.push_back(timing.inference_ms);
      d2h.push_back(timing.d2h_ms);
      gpu_pipeline.push_back(timing.gpu_pipeline_ms());
      backend_wall.push_back(timing.backend_wall_ms);
      postprocess.push_back(timing.postprocess_ms);
      total.push_back(timing.total_ms);
    }
  };

  Statistics summarize(std::vector<double> values)
  {
    if (values.empty()) throw std::invalid_argument("Cannot summarize empty samples");
    std::sort(values.begin(), values.end());
    const auto percentile = [&values](double p)
    {
      const std::size_t index = static_cast<std::size_t>(
          std::ceil(p * static_cast<double>(values.size()))) - 1;
      return values[std::min(index, values.size() - 1)];
    };
    Statistics stats;
    stats.mean = std::accumulate(values.begin(), values.end(), 0.0) /
                 static_cast<double>(values.size());
    stats.minimum = values.front();
    stats.p50 = percentile(0.50);
    stats.p95 = percentile(0.95);
    return stats;
  }

  bool same_boxes(const std::vector<lite::types::Boxf> &expected,
                  const std::vector<lite::types::Boxf> &actual,
                  float tolerance = 1e-3f)
  {
    if (expected.size() != actual.size()) return false;
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
      if (expected[i].label != actual[i].label ||
          std::fabs(expected[i].score - actual[i].score) > tolerance ||
          std::fabs(expected[i].x1 - actual[i].x1) > tolerance ||
          std::fabs(expected[i].y1 - actual[i].y1) > tolerance ||
          std::fabs(expected[i].x2 - actual[i].x2) > tolerance ||
          std::fabs(expected[i].y2 - actual[i].y2) > tolerance)
        return false;
    }
    return true;
  }

  int positive_integer(const char *value, const char *name)
  {
    const int parsed = std::stoi(value);
    if (parsed <= 0)
      throw std::invalid_argument(std::string(name) + " must be positive");
    return parsed;
  }

  void run_once(Detector &detector, const cv::Mat &image, PipelineMode mode,
                std::vector<lite::types::Boxf> &boxes, Timing *timing)
  {
    if (timing)
      detector.detect_with_timing(image, boxes, *timing, 0.25f, 100, mode);
    else
      detector.detect(image, boxes, 0.25f, 100, mode);
  }

  void print_metric(const char *mode, const char *name,
                    const std::vector<double> &samples)
  {
    const Statistics stats = summarize(samples);
    std::cout << mode << ',' << name << ',' << stats.mean << ',' << stats.p50 << ','
              << stats.p95 << ',' << stats.minimum << '\n';
  }

  void print_samples(const char *mode, const Samples &samples)
  {
    print_metric(mode, "preprocess_cpu", samples.preprocess);
    print_metric(mode, "h2d_gpu", samples.h2d);
    print_metric(mode, "preprocess_gpu", samples.gpu_preprocess);
    print_metric(mode, "inference_gpu", samples.inference);
    print_metric(mode, "d2h_gpu", samples.d2h);
    print_metric(mode, "gpu_pipeline", samples.gpu_pipeline);
    print_metric(mode, "backend_wall", samples.backend_wall);
    print_metric(mode, "postprocess_cpu", samples.postprocess);
    print_metric(mode, "total_wall", samples.total);
  }

  void print_comparison(const char *comparison, const char *name,
                        const std::vector<double> &reference,
                        const std::vector<double> &candidate)
  {
    const double reference_mean = summarize(reference).mean;
    const double candidate_mean = summarize(candidate).mean;
    const double delta = candidate_mean - reference_mean;
    std::cout << comparison << ',' << name << ',' << reference_mean << ','
              << candidate_mean << ',' << delta << ',';
    if (reference_mean == 0.0)
      std::cout << "N/A\n";
    else
      std::cout << delta * 100.0 / reference_mean << '\n';
  }

  void print_comparisons(const char *comparison, const Samples &reference,
                         const Samples &candidate)
  {
    print_comparison(comparison, "preprocess_cpu", reference.preprocess, candidate.preprocess);
    print_comparison(comparison, "h2d_gpu", reference.h2d, candidate.h2d);
    print_comparison(comparison, "preprocess_gpu", reference.gpu_preprocess,
                     candidate.gpu_preprocess);
    print_comparison(comparison, "inference_gpu", reference.inference, candidate.inference);
    print_comparison(comparison, "d2h_gpu", reference.d2h, candidate.d2h);
    print_comparison(comparison, "gpu_pipeline", reference.gpu_pipeline,
                     candidate.gpu_pipeline);
    print_comparison(comparison, "backend_wall", reference.backend_wall,
                     candidate.backend_wall);
    print_comparison(comparison, "postprocess_cpu", reference.postprocess,
                     candidate.postprocess);
    print_comparison(comparison, "total_wall", reference.total, candidate.total);
  }

  void print_preprocess_comparison(const char *layout,
                                   const Detector::PreprocessComparison &comparison)
  {
    std::cout << "preprocess_validation," << layout << ',' << comparison.elements << ','
              << comparison.mismatched << ',' << comparison.mean_abs_error << ','
              << comparison.p99_abs_error << ',' << comparison.max_abs_error << '\n';
  }

  void require_exact_preprocess(Detector &detector, const std::string &name,
                                const cv::Mat &image)
  {
    const Detector::PreprocessComparison contiguous = detector.compare_preprocess(image);
    print_preprocess_comparison((name + "_contiguous").c_str(), contiguous);
    if (contiguous.mismatched != 0 || contiguous.max_abs_error != 0.0f)
      throw std::runtime_error("Fused preprocess differs from OpenCV for " + name);

    if (image.rows == 1) return;

    cv::Mat storage(image.rows + 4, image.cols + 6, image.type());
    cv::Mat roi = storage(cv::Rect(3, 2, image.cols, image.rows));
    image.copyTo(roi);
    if (roi.isContinuous())
      throw std::runtime_error("Internal non-contiguous ROI validation setup failed");
    const Detector::PreprocessComparison stepped = detector.compare_preprocess(roi);
    print_preprocess_comparison((name + "_non_contiguous_roi").c_str(), stepped);
    if (stepped.mismatched != 0 || stepped.max_abs_error != 0.0f)
      throw std::runtime_error("Fused preprocess differs from OpenCV for stepped " + name);
  }

  cv::Mat make_pattern(int rows, int cols)
  {
    cv::Mat image(rows, cols, CV_8UC3);
    for (int y = 0; y < rows; ++y)
    {
      cv::Vec3b *row = image.ptr<cv::Vec3b>(y);
      for (int x = 0; x < cols; ++x)
      {
        row[x][0] = static_cast<unsigned char>((x * 17 + y * 29 + 3) & 255);
        row[x][1] = static_cast<unsigned char>((x * 7 + y * 13 + 91) & 255);
        row[x][2] = static_cast<unsigned char>((x * 31 + y * 5 + 47) & 255);
      }
    }
    return image;
  }

  std::array<PipelineMode, 3> balanced_order(int iteration)
  {
    static const std::array<std::array<PipelineMode, 3>, 6> orders = {{
        {{PipelineMode::Baseline, PipelineMode::PinnedCpu, PipelineMode::Optimized}},
        {{PipelineMode::Optimized, PipelineMode::Baseline, PipelineMode::PinnedCpu}},
        {{PipelineMode::PinnedCpu, PipelineMode::Baseline, PipelineMode::Optimized}},
        {{PipelineMode::Optimized, PipelineMode::PinnedCpu, PipelineMode::Baseline}},
        {{PipelineMode::Baseline, PipelineMode::Optimized, PipelineMode::PinnedCpu}},
        {{PipelineMode::PinnedCpu, PipelineMode::Optimized, PipelineMode::Baseline}}
    }};
    return orders[static_cast<std::size_t>(iteration) % orders.size()];
  }

  void run_mode(Detector &detector, const cv::Mat &image, PipelineMode mode,
                std::vector<lite::types::Boxf> &baseline_boxes,
                std::vector<lite::types::Boxf> &previous_boxes,
                std::vector<lite::types::Boxf> &optimized_boxes,
                Timing *baseline_timing, Timing *previous_timing,
                Timing *optimized_timing)
  {
    if (mode == PipelineMode::Baseline)
      run_once(detector, image, mode, baseline_boxes, baseline_timing);
    else if (mode == PipelineMode::PinnedCpu)
      run_once(detector, image, mode, previous_boxes, previous_timing);
    else
      run_once(detector, image, mode, optimized_boxes, optimized_timing);
  }

  int run_benchmark(int argc, char *argv[])
  {
    if (argc < 4 || argc > 6)
    {
      std::cerr << "Usage: " << argv[0]
                << " --benchmark <engine_path> <image_path> [warmup=20] [iterations=200]"
                << std::endl;
      return EXIT_FAILURE;
    }

    try
    {
      const std::string engine_path = argv[2];
      const std::string image_path = argv[3];
      const int warmup = argc >= 5 ? positive_integer(argv[4], "warmup") : 20;
      const int iterations = argc >= 6 ? positive_integer(argv[5], "iterations") : 200;

      cv::Mat image = cv::imread(image_path);
      if (image.empty())
        throw std::runtime_error("Failed to read benchmark image: " + image_path);

      Detector detector(engine_path);
      std::cout << std::fixed << std::setprecision(6);
      std::cout << "preprocess_validation,layout,elements,mismatched_exact,mean_abs_error,p99_abs_error,max_abs_error\n";
      require_exact_preprocess(detector, "benchmark_image", image);
      require_exact_preprocess(detector, "upscale_319x511", make_pattern(319, 511));
      require_exact_preprocess(detector, "downscale_721x1283", make_pattern(721, 1283));
      require_exact_preprocess(detector, "square_640x640", make_pattern(640, 640));
      require_exact_preprocess(detector, "single_row_1x97", make_pattern(1, 97));

      std::vector<lite::types::Boxf> reference_boxes;
      std::vector<lite::types::Boxf> baseline_boxes;
      std::vector<lite::types::Boxf> previous_boxes;
      std::vector<lite::types::Boxf> optimized_boxes;

      for (int i = 0; i < warmup; ++i)
      {
        for (const PipelineMode mode : balanced_order(i))
          run_mode(detector, image, mode, baseline_boxes, previous_boxes,
                   optimized_boxes, nullptr, nullptr, nullptr);
        if (!same_boxes(baseline_boxes, previous_boxes) ||
            !same_boxes(baseline_boxes, optimized_boxes))
          throw std::runtime_error(
              "Baseline, previous, and optimized detections differ during warmup");
        reference_boxes = baseline_boxes;
      }

      Samples baseline;
      Samples previous;
      Samples optimized;
      baseline.reserve(static_cast<std::size_t>(iterations));
      previous.reserve(static_cast<std::size_t>(iterations));
      optimized.reserve(static_cast<std::size_t>(iterations));

      for (int i = 0; i < iterations; ++i)
      {
        Timing baseline_timing;
        Timing previous_timing;
        Timing optimized_timing;
        for (const PipelineMode mode : balanced_order(i))
          run_mode(detector, image, mode, baseline_boxes, previous_boxes,
                   optimized_boxes, &baseline_timing, &previous_timing,
                   &optimized_timing);

        if (!same_boxes(reference_boxes, baseline_boxes) ||
            !same_boxes(reference_boxes, previous_boxes) ||
            !same_boxes(reference_boxes, optimized_boxes))
          throw std::runtime_error("Detection output changed during three-way benchmark");

        baseline.append(baseline_timing);
        previous.append(previous_timing);
        optimized.append(optimized_timing);
      }

      const double baseline_total = summarize(baseline.total).mean;
      const double previous_total = summarize(previous.total).mean;
      const double optimized_total = summarize(optimized.total).mean;
      const double baseline_fps = 1000.0 / baseline_total;
      const double previous_fps = 1000.0 / previous_total;
      const double optimized_fps = 1000.0 / optimized_total;

      std::cout << std::fixed << std::setprecision(6);
      std::cout << "engine," << engine_path << '\n';
      std::cout << "image," << image_path << '\n';
      std::cout << "warmup_sets," << warmup << '\n';
      std::cout << "measured_sets," << iterations << '\n';
      std::cout << "boxes," << reference_boxes.size() << '\n';
      std::cout << "consistency_checks," << iterations << '\n';
      std::cout << "consistency,passed\n";
      std::cout << "mode,metric,mean_ms,p50_ms,p95_ms,min_ms\n";
      print_samples("baseline", baseline);
      print_samples("previous_pinned_cpu", previous);
      print_samples("optimized_cuda", optimized);
      std::cout << "mode,throughput_fps\n";
      std::cout << "baseline," << baseline_fps << '\n';
      std::cout << "previous_pinned_cpu," << previous_fps << '\n';
      std::cout << "optimized_cuda," << optimized_fps << '\n';
      std::cout << "comparison,metric,reference_mean_ms,optimized_mean_ms,delta_ms,delta_percent\n";
      print_comparisons("from_baseline", baseline, optimized);
      print_comparisons("from_previous", previous, optimized);
      std::cout << "speedup_from_baseline," << baseline_total / optimized_total << '\n';
      std::cout << "speedup_from_previous," << previous_total / optimized_total << '\n';
      std::cout << "fps_delta_from_baseline," << optimized_fps - baseline_fps << '\n';
      std::cout << "fps_delta_from_previous," << optimized_fps - previous_fps << '\n';
      return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
      std::cerr << "Benchmark failed: " << error.what() << std::endl;
      return EXIT_FAILURE;
    }
  }
}
#endif

int main(int argc, char *argv[])
{
  if (argc > 1 && std::string(argv[1]) == "--benchmark")
  {
#ifdef ENABLE_TENSORRT
    return run_benchmark(argc, argv);
#else
    std::cerr << "This benchmark requires ENABLE_TENSORRT=ON" << std::endl;
    return EXIT_FAILURE;
#endif
  }

  test_lite();
  return EXIT_SUCCESS;
}
