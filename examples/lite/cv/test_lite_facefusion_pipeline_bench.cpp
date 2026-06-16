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

  // Decode the two images ONCE, outside the timed loop. Real pipelines (video /
  // server) decode at the edge, not per frame; keeping imread/imwrite out of the
  // loop is what makes this a *compute-only* benchmark (the file-path detect()
  // overload would re-read both images and write the result every iteration).
  cv::Mat src = cv::imread(source_img);
  cv::Mat tgt = cv::imread(target_img);
  if (src.empty() || tgt.empty()) {
    std::cerr << "[bench] cannot read source/target image" << std::endl;
    return 1;
  }
  std::cout << "[bench] source=" << source_img << " target=" << target_img
            << "\n[bench] warmup=" << warmup << " iters=" << iters
            << "  (compute-only, video-style: prepare_source once + per-frame process)" << std::endl;

  // Video / server use case: the SOURCE face is fixed, so prepare it ONCE and then time only
  // the per-frame process(target). This is what the source-embedding cache buys — the loop no
  // longer re-runs detect_src / landmark_src / recognizer every frame.
  pipeline.prepare_source(src, 0);

  // Warmup (lazy engine/context init, cudnn autotune) — excluded from stats.
  for (int i = 0; i < warmup; ++i)
    pipeline.process(tgt, 0);

  lite::bench::Profiler prof;
  cv::Mat out;
  for (int i = 0; i < iters; ++i) {
    // Sanity: GPU memory should stay flat across iterations (no leak / no
    // per-call buffer growth). Printed sparsely to avoid flooding the output.
    if (i == 0 || i == iters - 1 || i % 10 == 0) {
      size_t freeB = 0, totalB = 0;
      cudaMemGetInfo(&freeB, &totalB);
      std::cout << "[mem] iter " << i << " used=" << (totalB - freeB) / (1024 * 1024)
                << " MiB" << std::endl;
    }
    lite::bench::CpuTimer t;
    t.start();
    out = pipeline.process(tgt, 0, &prof);
    prof.tick(t.stop_ms());
  }

  prof.report("FaceFusion pipeline (per-frame, source cached)");
  prof.to_csv(csv_path);
  if (!out.empty()) {
    cv::imwrite(out_path, out);   // save one result (outside the timed loop) for visual check
    std::cout << "[bench] sample result: " << out_path << std::endl;
  }
  return 0;
}
#else
int main() {
  std::cerr << "This benchmark requires ENABLE_TENSORRT=ON.\n";
  return 0;
}
#endif
