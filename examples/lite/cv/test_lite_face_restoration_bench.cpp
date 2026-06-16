//
// End-to-end benchmark for the GFPGAN face-restoration stage (Phase 0).
// Usage:
//   lite_face_restoration_bench [engine_path] [test_img] [iters] [warmup] [csv]
// Defaults point at a gfpgan engine + a test image on the remote 4090; override via argv.
//
// It first runs a CPU-vs-GPU paste_back equivalence check, then a compute-only
// latency/throughput benchmark of restore() with per-stage aggregation
// (preprocess / infer / postprocess / paste_back). Disk I/O (imwrite) is kept
// out of the timed loop; one result image is saved afterwards for visual checking.
//
#include "lite/lite.h"
#include "lite/bench/profiler.h"

#ifdef ENABLE_TENSORRT
#include "lite/trt/cv/trt_face_restoration.h"
#include "lite/ort/cv/face_utils.h"
#include "lite/trt/kernel/paste_back_manager.h"

// A/B numerical check: per-pixel difference between the GPU fused paste_back and the
// CPU reference, on the same real inputs (real affine + real crop).
static void check_paste_back_equivalence(const cv::Mat &frame,
                                         std::vector<cv::Point2f> &lmk5) {
  // Run the real warp to obtain the real affine + real 512 crop (uint8), then to float (as in the pipeline)
  cv::Mat crop_u8, affine;
  std::tie(crop_u8, affine) =
      face_utils::warp_face_by_face_landmark_5(frame, lmk5, face_utils::FFHQ_512);
  cv::Mat crop_f;
  crop_u8.convertTo(crop_f, CV_32FC3);
  cv::Mat mask = face_utils::create_static_box_mask({512, 512});

  cv::Mat out_cpu = launch_paste_back(frame, crop_f, mask, affine);
  PasteBackGPU gpu;
  cv::Mat out_gpu = gpu.paste_back(frame, crop_f, mask, affine, nullptr);

  cv::Mat diff;
  cv::absdiff(out_cpu, out_gpu, diff);
  cv::Scalar mean_diff = cv::mean(diff);
  double max_diff = 0.0;
  cv::minMaxLoc(diff.reshape(1), nullptr, &max_diff);
  std::cout << "[check] paste_back CPU vs GPU  max|diff|=" << max_diff
            << "  mean|diff|(B,G,R)=" << mean_diff[0] << "," << mean_diff[1]
            << "," << mean_diff[2] << "  (uint8 pixel values, smaller = closer)" << std::endl;
}
#endif

int main(__unused int argc, __unused char *argv[]) {
#ifdef ENABLE_TENSORRT
  std::string engine_path =
      argc > 1 ? argv[1] : "/root/autodl-tmp/gfpgan_proj/gfpgan_fp32.engine";
  std::string test_img_path =
      argc > 2 ? argv[2] : "../../../examples/lite/resources/test_lite_face_restoration.jpg";
  int iters = argc > 3 ? std::atoi(argv[3]) : 50;
  int warmup = argc > 4 ? std::atoi(argv[4]) : 10;
  std::string csv_path = argc > 5 ? argv[5] : "bench_face_restoration.csv";

  // Fixed 5-point landmarks (same as test_lite_face_restoration.cpp); the benchmark only
  // cares about timing, so whether they exactly match the image does not affect the numbers.
  std::vector<cv::Point2f> face_landmark_5 = {
      cv::Point2f(569.092041f, 398.845886f),
      cv::Point2f(701.891724f, 399.156677f),
      cv::Point2f(634.767212f, 482.927216f),
      cv::Point2f(584.270996f, 543.294617f),
      cv::Point2f(684.877991f, 543.067078f)};

  cv::Mat img_bgr = cv::imread(test_img_path);
  if (img_bgr.empty()) {
    std::cerr << "[bench] cannot read test image: " << test_img_path << std::endl;
    return 1;
  }

  std::cout << "[bench] engine=" << engine_path << "\n[bench] img="
            << test_img_path << " (" << img_bgr.cols << "x" << img_bgr.rows
            << ")\n[bench] warmup=" << warmup << " iters=" << iters << std::endl;

  // Numerical correctness check (CPU vs GPU paste_back) before benchmarking
  check_paste_back_equivalence(img_bgr, face_landmark_5);

  trtcv::TRTFaceFusionFaceRestoration restorer(engine_path);
  const std::string tmp_out = "/tmp/bench_restoration_out.jpg";

  // Warmup (first runs include lazy engine/context init and cudnn autotune; excluded from stats)
  for (int i = 0; i < warmup; ++i) {
    restorer.restore(img_bgr, face_landmark_5, nullptr);
  }

  // Timed: restore() does not write to disk; the profiler collects per-stage timings
  // (preprocess/infer/postprocess/paste_back) and the end-to-end TOTAL. imwrite is moved
  // out of the loop; one image is saved at the end for visual verification.
  lite::bench::Profiler prof;
  cv::Mat dst;
  for (int i = 0; i < iters; ++i) {
    lite::bench::CpuTimer t;
    t.start();
    dst = restorer.restore(img_bgr, face_landmark_5, &prof);
    prof.tick(t.stop_ms());
  }

  prof.report("GFPGAN face restoration (compute-only, no disk I/O)");
  prof.to_csv(csv_path);

  if (!dst.empty()) {
    cv::imwrite(tmp_out, dst);
    std::cout << "[bench] sample result (saved once, outside the loop): " << tmp_out << std::endl;
  }
#else
  std::cerr << "This benchmark requires ENABLE_TENSORRT=ON." << std::endl;
#endif
  return 0;
}
