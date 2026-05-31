//
// FaceFusion face-swap pipeline — out-of-box CLI runner (TensorRT).
//
// Unlike the other examples, this one takes every path from argv so you can run
// the flagship pipeline on your own images without editing/recompiling source.
//
//   lite_facefusion_cli <engine_dir> <source_img> <target_img> <output_img> [src_face_idx] [tgt_face_idx]
//
// `engine_dir` must contain the 5 TensorRT engines (build them once from ONNX with
// ./build_facefusion_engines.sh). See docs/facefusion_quickstart.md.
//
#include "lite/lite.h"
#include <iostream>
#include <string>

// Default engine filenames expected inside <engine_dir>.
static const char *kFaceDetectEngine     = "yoloface_8n_fp16.engine";
static const char *kFaceLandmarksEngine  = "2dfan4_fp16.engine";
static const char *kFaceRecognizerEngine = "arcface_w600k_r50_fp16.engine";
static const char *kFaceSwapEngine       = "inswapper_128_fp16.engine";
static const char *kFaceRestoreEngine    = "gfpgan_1.4_fp32.engine";

static void usage(const char *prog)
{
  std::cout
      << "Usage: " << prog
      << " <engine_dir> <source_img> <target_img> <output_img> [src_face_idx=0] [tgt_face_idx=0]\n\n"
      << "  engine_dir  directory holding the 5 TensorRT engines:\n"
      << "                " << kFaceDetectEngine << ", " << kFaceLandmarksEngine << ",\n"
      << "                " << kFaceRecognizerEngine << ", " << kFaceSwapEngine << ", "
      << kFaceRestoreEngine << "\n"
      << "  source_img  image whose face is taken\n"
      << "  target_img  image whose face is replaced\n"
      << "  output_img  where to write the swapped result\n";
}

int main(int argc, char *argv[])
{
#ifdef ENABLE_TENSORRT
  if (argc < 5)
  {
    usage(argv[0]);
    return 1;
  }
  const std::string engine_dir = argv[1];
  const std::string source_img = argv[2];
  const std::string target_img = argv[3];
  const std::string output_img = argv[4];
  const int src_idx = (argc > 5) ? std::stoi(argv[5]) : 0;
  const int tgt_idx = (argc > 6) ? std::stoi(argv[6]) : 0;

  const std::string sep =
      (engine_dir.empty() || engine_dir.back() == '/') ? "" : "/";
  auto engine = [&](const char *name) { return engine_dir + sep + name; };

  auto pipeline = lite::trt::cv::face::swap::FaceFusionPipeLine(
      engine(kFaceDetectEngine),
      engine(kFaceLandmarksEngine),
      engine(kFaceRecognizerEngine),
      engine(kFaceSwapEngine),
      engine(kFaceRestoreEngine));

  pipeline.detect(source_img, src_idx, target_img, tgt_idx, output_img);
  std::cout << "[FaceFusion] wrote: " << output_img << std::endl;
  return 0;
#else
  (void) argc;
  (void) argv;
  std::cerr << "This binary needs the TensorRT backend. Rebuild with: bash ./build.sh tensorrt\n";
  return 1;
#endif
}
