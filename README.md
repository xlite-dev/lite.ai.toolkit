
<div id="lite.ai.toolkit-Introduction"></div>

![lite-ai-toolkit](https://github.com/user-attachments/assets/11568474-57e3-4ef7-96c0-d2ce7028bb5f)

<div align='center'>
  <img src=https://img.shields.io/badge/Linux-pass-brightgreen.svg >
  <img src=https://img.shields.io/badge/Device-GPU-yellow.svg >
  <img src=https://img.shields.io/badge/TensorRT-10-turquoise.svg >
  <img src=https://img.shields.io/badge/CUDA-12-turquoise.svg >
  <img src=https://img.shields.io/badge/ONNXRuntime-1.17.1-turquoise.svg >
  <img src=https://img.shields.io/github/stars/xlite-dev/lite.ai.toolkit.svg?style=social >
</div>

🛠 **Lite.Ai.ToolKit** is a C++ toolkit for **extreme GPU inference**. The flagship is an end-to-end
**FaceFusion face-swap pipeline** (detect → landmark → recognize → swap → restore) running entirely on
**TensorRT**, with the CPU pre/post-processing glue rewritten as **hand-fused CUDA kernels**. The goal is
not breadth — it is to make one real pipeline as fast as a single GPU can make it, and to show the work
honestly with a reproducible benchmark harness. Welcome to 🌟 star this repo to support us ~ 🎉🎉

> **Heads up (>= 0.3):** the active line targets **TensorRT only**. ONNXRuntime is kept as the numerical
> reference + the host for the test suite. The legacy multi-backend build (MNN / NCNN / TNN, 300+ thin
> model wrappers) is frozen on tag **[`v0.2-all-backends`](https://github.com/xlite-dev/lite.ai.toolkit/tree/main)** — check it out if you need those backends.

## 📖 News 🔥🔥
<div id="news"></div>

- **GPU-inference optimization in progress** — the FaceFusion face-restoration stage (GFPGAN 1.4) was
  taken from **78.2 ms → 17.7 ms (4.4×, 12.8 → 56.6 FPS)** on an RTX 4090 by moving paste-back and
  preprocessing into fused CUDA kernels. See [Benchmark](#benchmark) below. The rest of the pipeline
  (detect / landmark / swap, FP16) is being optimized stage by stage.
- [lite.ai.toolkit](https://github.com/xlite-dev/lite.ai.toolkit) is mainly maintained by 🎉[@wangzijian1010](https://github.com/wangzijian1010).

## ⚡ Benchmark 🔥
<div id="benchmark"></div>

GPU-inference optimization log. For each algorithm we profile it with a built-in, backend-agnostic harness ([`lite/bench/profiler.h`](https://github.com/xlite-dev/lite.ai.toolkit/blob/main/lite/bench/profiler.h)), then move the CPU pre/post-processing (affine warp, color convert, normalize, tensor layout, paste-back, NMS …) into **fused CUDA kernels** with reused device buffers and pinned + async copies, so the algorithm spends its time on real inference instead of host glue and `cudaMalloc`/sync round-trips. All numbers are **RTX 4090 · TensorRT 10.1 · CUDA 12.4**, median (p50), compute-only, reproducible via the `lite_*_bench` binaries.

| Algorithm | Before | After | Speedup | What changed |
|:--|:--:|:--:|:--:|:--|
| **FaceFusion · face restoration (GFPGAN 1.4)** | 78.2 ms<br>(12.8 FPS) | **17.7 ms<br>(56.6 FPS)** | **4.4×** | inverse-mapping paste-back kernel (replaces 2× CPU `warpAffine` + per-frame `cudaMalloc`); cached static mask; fused `bgr2rgb+normalize+CHW` straight into the input buffer |
| FaceFusion · face detect (YOLOv8-face) | 🚧 | 🚧 | — | bbox decode + NMS → CUDA |
| FaceFusion · 68 landmarks (2DFAN4) | 🚧 | 🚧 | — | warp + preprocess → CUDA |
| FaceFusion · face swap (InSwapper) | 🚧 | 🚧 | — | warp + paste → CUDA |
| FP16 / mixed-precision | 🚧 | 🚧 | — | layer-pinned style convs (keep sensitive layers FP32) |

<details>
<summary><b>FaceFusion · face restoration — per-stage breakdown</b></summary>

| Stage | Baseline (ms) | Optimized (ms) | Speedup |
|:--|:--:|:--:|:--:|
| preprocess (warp + bgr2rgb + normalize + tensor) | 14.49 | 1.25 | **11.6×** |
| inference (TensorRT) | 11.30 | 10.79 | 1.05× |
| postprocess (incl. paste-back) | 52.02 | 5.32 | **9.8×** |
| &nbsp;&nbsp;└ paste-back | 39.07 | 2.34 | **16.7×** |
| **End-to-end** | **78.17** | **17.66** | **4.4×** |

paste-back is numerically equivalent to the CPU path (max |diff| = 2/255). The static box mask used to be rebuilt every frame (a large-kernel Gaussian blur) although it only depends on the crop size. With pre/post off the critical path, inference is now ~60% of the stage — FP16 is the next lever.

</details>

## Features 👏👋

- **GPU-first.** The whole FaceFusion pipeline runs on TensorRT; the pre/post-processing that usually
  lingers on the CPU (warp / color-convert / normalize / layout / paste-back / NMS) is implemented as
  **fused CUDA kernels** under [`lite/trt/kernel/`](https://github.com/xlite-dev/lite.ai.toolkit/tree/main/lite/trt/kernel), with reused device buffers and pinned + async copies.
- **Measured, not claimed.** A header-only profiler ([`lite/bench/profiler.h`](https://github.com/xlite-dev/lite.ai.toolkit/blob/main/lite/bench/profiler.h)) gives CPU-chrono + GPU-cudaEvent timings (p50 / p99 / FPS / CSV). Every optimization ships with a before/after `lite_*_bench` binary.
- **Multi-threaded TRT path.** `_mt` pipelines (e.g. `trt_face_restoration_mt`) run a thread pool with one
  `IExecutionContext` + `cudaStream_t` + buffer set per thread and an async task queue.
- **Consistent C++ API.** Same `lite::trt::cv::Type::Class` syntax across models, e.g. `lite::trt::cv::detection::YOLOV5`.

## Build 👇👇

TensorRT is the maintained backend. It needs **TensorRT 10.x** and **CUDA 12.x** (Linux only). The first
build downloads third-party libs into `third_party/` automatically.

```shell
git clone --depth=1 https://github.com/xlite-dev/lite.ai.toolkit.git
cd lite.ai.toolkit
bash ./build.sh tensorrt   # GPU / TensorRT backend
# bash ./build.sh          # ONNXRuntime backend (CPU reference + 100+ CV models, builds the tests)
```

See [tensorrt-linux-x86_64.zh.md](./docs/tensorrt/tensorrt-linux-x86_64.zh.md) for the TensorRT/CUDA setup.

## Quick Start 🌟🌟
<div id="lite.ai.toolkit-Quick-Start"></div>

#### Flagship: FaceFusion face-swap pipeline on the GPU
End-to-end source→target face swap, fully on TensorRT. See [`test_lite_facefusion_pipeline.cpp`](https://github.com/xlite-dev/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_facefusion_pipeline.cpp) for the full example (engine paths + I/O).

```c++
#include "lite/lite.h"
// build the 5 engines once, e.g. trtexec --onnx=gfpgan_1.4.onnx --saveEngine=gfpgan_1.4_fp32.engine
auto pipeline = lite::trt::cv::face::swap::FaceFusionPipeLine(
    face_detect_engine,        // yoloface_8n
    face_landmarks_68_engine,  // 2dfan4
    face_recognizer_engine,    // arcface_w600k_r50
    face_swap_engine,          // inswapper_128
    face_restoration_engine);  // gfpgan_1.4
// swap face #0 of the source onto face #0 of the target, then write the result
pipeline.detect(source_image_path, 0, target_image_path, 0, save_image_path);
```

#### Single model on the GPU (YOLOv5)
```c++
#include "lite/lite.h"
// trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s.engine
auto *yolov5 = new lite::trt::cv::detection::YOLOV5(engine_path);
std::vector<lite::types::Boxf> boxes;
cv::Mat img = cv::imread(test_img_path);
yolov5->detect(img, boxes);
lite::utils::draw_boxes_inplace(img, boxes);
cv::imwrite(save_img_path, img);
delete yolov5;
```

## Quick Setup 👀

To use the installed library from your own project, point `find_package` at the install dir:

```cmake
set(lite.ai.toolkit_DIR YOUR-PATH-TO-LITE-INSTALL)
find_package(lite.ai.toolkit REQUIRED PATHS ${lite.ai.toolkit_DIR})
add_executable(lite_yolov5 test_lite_yolov5.cpp)
target_link_libraries(lite_yolov5 ${lite.ai.toolkit_LIBS})
```

## Supported Models (TensorRT) 🚀
<div id="lite.ai.toolkit-Supported-Models-Matrix"></div>

|Class|Class|Class|Class|Class| System | Engine |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|✅[YOLOv5](https://github.com/xlite-dev/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov5.cpp)|✅[YOLOv6](https://github.com/xlite-dev/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov6.cpp)|✅[YOLOv8](https://github.com/xlite-dev/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov8.cpp)|✅[YOLOv8Face](https://github.com/xlite-dev/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov8face.cpp)|✅[YOLOv5Face](https://github.com/xlite-dev/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolo5face.cpp)| Linux | TensorRT |
|✅[YOLOX](https://github.com/xlite-dev/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolox.cpp)|✅[YOLOv5BlazeFace](https://github.com/xlite-dev/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov5_blazeface.cpp)|✅[StableDiffusion](https://github.com/xlite-dev/lite.ai.toolkit/blob/main/examples/lite/sd/test_lite_sd_pipeline.cpp)|✅[FaceFusion](https://github.com/xlite-dev/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_facefusion_pipeline.cpp)| / | Linux | TensorRT |

> Also includes **100+ CPU / ONNXRuntime CV models** (detection, face recognition, segmentation, matting,
> classification, …) behind the same `lite::cv::Type::Class` API. They are not the focus of the active
> line but remain available — see the [ONNX Hub](https://github.com/xlite-dev/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md) for the full catalog and weights, or tag [`v0.2-all-backends`](https://github.com/xlite-dev/lite.ai.toolkit/tree/main) for the legacy multi-backend matrix.

## Architecture 🧩

```
lite/
├── trt/          # TensorRT backend — the maintained high-performance path
│   ├── core/     # trt_handler base (engine load, buffers, streams)
│   ├── cv/       # one .h/.cpp per model + the facefusion pipeline (+ _mt variants)
│   ├── kernel/   # hand-written fused CUDA kernels (.cu/.cuh) + host-side managers
│   └── sd/       # Stable Diffusion components (clip / unet / vae / scheduler)
├── ort/          # ONNXRuntime backend — numerical reference + test host (100+ CV models)
├── bench/        # header-only profiler (CPU chrono + GPU cudaEvent, p50/p99/FPS/CSV)
└── lite.h        # single public include
```

`lite::cv` is a compile-time namespace alias resolved in [`lite/models.h`](https://github.com/xlite-dev/lite.ai.toolkit/blob/main/lite/models.h). Pin a backend explicitly with `lite::trt::cv::...` (GPU) or `lite::onnxruntime::cv::...` (CPU reference).

## Citations 🎉🎉
```BibTeX
@misc{lite.ai.toolkit@2021,
  title={lite.ai.toolkit: A lite C++ toolkit of 100+ Awesome AI models.},
  url={https://github.com/xlite-dev/lite.ai.toolkit},
  note={Open-source software available at https://github.com/xlite-dev/lite.ai.toolkit},
  author={xlite-dev, wangzijian1010 etc},
  year={2021}
}
```

## ©️License
GNU General Public License v3.0

## 🎉Contribute
Please consider ⭐ this repo if you like it, as it is the simplest way to support us.

<div align='center'>
<a href="https://star-history.com/#xlite-dev/lite.ai.toolkit&Date">
  <picture align='center'>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=xlite-dev/lite.ai.toolkit&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=xlite-dev/lite.ai.toolkit&type=Date" />
    <img width=450 height=300 alt="Star History Chart" src="https://api.star-history.com/svg?repos=xlite-dev/lite.ai.toolkit&type=Date" />
  </picture>
</a>
</div>
</content>
