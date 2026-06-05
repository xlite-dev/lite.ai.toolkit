//
// Whole-pipeline benchmark for the FaceFusion face-swap pipeline.
//   lite_facefusion_pipeline_bench <detect_engine> <landmark68_engine> <recognizer_engine> \
//                                  <swap_engine> <restoration_engine> <source_img> <target_img> \
//                                  [iters=30] [warmup=5] [csv]
//
// Profiles the pipeline into per-stage times (imread / detect / landmark — each x2 for
// source+target — / recognizer / swap / restoration) so we can see where the end-to-end
// time actually goes BEFORE optimizing anything. Each stage returns a host-visible result,
// so CPU-side timing is accurate.
//
// NOTE: the "restoration" stage here includes the final imwrite (the pipeline writes the
// result to disk), so it reads a bit higher than the compute-only restoration bench.
//
#include "lite/lite.h"
#include "lite/bench/profiler.h"
#include <iostream>
#include <string>
#include <cstdlib>

#ifdef ENABLE_TENSORRT
int main(int argc, char *argv[]) {
  if (argc < 8) {
    std::cout << "Usage: " << argv[0]
              << " <detect_engine> <landmark68_engine> <recognizer_engine> <swap_engine>"
                 " <restoration_engine> <source_img> <target_img> [iters=30] [warmup=5] [csv]\n";
    return 1;
  }
  const std::string detect_engine      = argv[1];
  const std::string landmark_engine    = argv[2];
  const std::string recognizer_engine  = argv[3];
  const std::string swap_engine        = argv[4];
  const std::string restoration_engine = argv[5];
  const std::string source_img         = argv[6];
  const std::string target_img         = argv[7];
  const int iters   = argc > 8 ? std::atoi(argv[8]) : 30;
  const int warmup  = argc > 9 ? std::atoi(argv[9]) : 5;
  const std::string csv_path = argc > 10 ? argv[10] : "bench_facefusion_pipeline.csv";

  const std::string out_path = "/tmp/bench_facefusion_out.jpg";

  lite::trt::cv::face::swap::FaceFusionPipeLine pipeline(
      detect_engine, landmark_engine, recognizer_engine, swap_engine, restoration_engine);

  std::cout << "[bench] source=" << source_img << " target=" << target_img
            << "\n[bench] warmup=" << warmup << " iters=" << iters << std::endl;

  // Warmup (lazy engine/context init, cudnn autotune) — excluded from stats.
  for (int i = 0; i < warmup; ++i)
    pipeline.detect(source_img, 0, target_img, 0, out_path);

  lite::bench::Profiler prof;
  for (int i = 0; i < iters; ++i) {
    lite::bench::CpuTimer t;
    t.start();
    pipeline.detect(source_img, 0, target_img, 0, out_path, &prof);
    prof.tick(t.stop_ms());
  }

  prof.report("FaceFusion pipeline (per-stage, end-to-end)");
  prof.to_csv(csv_path);
  std::cout << "[bench] sample result: " << out_path << std::endl;
  return 0;
}
#else
int main() {
  std::cerr << "This benchmark requires ENABLE_TENSORRT=ON.\n";
  return 0;
}
#endif
