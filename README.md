
<div id="lite.ai.toolkit-Introduction"></div>

![lite-ai-toolkit](https://github.com/user-attachments/assets/11568474-57e3-4ef7-96c0-d2ce7028bb5f)

<div align='center'>
  <img src=https://img.shields.io/badge/Linux-pass-brightgreen.svg >
  <img src=https://img.shields.io/badge/Device-GPU-yellow.svg >
  <img src=https://img.shields.io/badge/TensorRT-10-turquoise.svg >
  <img src=https://img.shields.io/badge/CUDA-12-turquoise.svg >
  <img src=https://img.shields.io/github/stars/xlite-dev/lite.ai.toolkit.svg?style=social >
</div>

🛠 **Lite.Ai.ToolKit** is a C++ toolkit focused on one flagship target: an end-to-end
**FaceFusion face-swap pipeline** (detect → landmark → recognize → swap → restore) running on
**TensorRT**. The current line is about keeping the real pipeline GPU-resident, not collecting model
wrappers: CUDA / NPP kernels handle the hot pre/post-processing, `DeviceFrame` carries full frames
between stages, and the benchmark reports the real per-frame path.

> **Heads up (>= 0.3):** the active line targets **TensorRT only**. ONNXRuntime is kept as the numerical
> reference + the host for the test suite. The legacy multi-backend build (MNN / NCNN / TNN, 300+ thin
> model wrappers) is frozen on tag **[`v0.2-all-backends`](https://github.com/xlite-dev/lite.ai.toolkit/tree/v0.2-all-backends)** — check it out if you need those backends.

## 📖 News 🔥🔥
<div id="news"></div>

- **Current FaceFusion pipeline:** **23.6 ms / frame, 42.3 FPS** on an RTX 4090, FP16 deployment,
  source prepared once and target processed per frame.
- **Current full-frame copies:** one H2D upload of the target frame, one D2H download of the final
  result. The swap → restoration boundary stays GPU-resident.
- Now, [lite.ai.toolkit](https://github.com/xlite-dev/lite.ai.toolkit) is mainly maintained by 🎉[@wangzijian1010](https://github.com/wangzijian1010).

## ⚡ Benchmark 🔥
<div id="benchmark"></div>

Measured on **RTX 4090 · TensorRT 10.x · CUDA 12.x**, FP16 deployment, source prepared once and
60 per-frame target iterations.

| Stage | Time |
|:--|--:|
| face detect | 3.94 ms |
| 68 landmarks | 3.46 ms |
| face swap | 4.96 ms |
| face restoration | 9.85 ms |
| **TOTAL** | **23.6 ms / frame** |
| **Throughput** | **42.3 FPS** |
| GPU memory | 1550 MiB |

## Data Flow

Current full-frame data movement is down to the intended minimum:

| Copy | Direction | Purpose |
|:--|:--|:--|
| 1 | Host → Device | upload target frame once into `target_dev_` |
| 2 | Device → Host | download final restored result |

The expensive swap → restoration boundary no longer bounces through host memory:

```text
target host Mat
  -> H2D once into target_dev_
  -> swap NPP warp + preprocess + infer + paste_back
  -> swapped_frame_ DeviceFrame
  -> restoration NPP warp + preprocess + infer + postprocess + paste_back
  -> D2H final result
```

Remaining copies are small: detect letterbox input/output metadata, landmark crop/output points, swap's
128 crop transpose bounce, and restoration mask/affine uploads.

## Headroom

Latency is now close to model-bound. The largest remaining block is GFPGAN inference inside restoration
(about 8 ms), so further single-frame latency gains are harder without quality-risky model changes such
as INT8 or a lighter restorer. The more realistic path toward 60+ FPS is throughput work: multi-stream
frame pipelining and CUDA Graphs, so different frames can overlap instead of running fully serial.

## Features 👏👋

- **GPU-first.** The whole FaceFusion pipeline runs on TensorRT; the pre/post-processing that usually
  lingers on the CPU (warp / color-convert / normalize / layout / paste-back / NMS) is being moved into
  **CUDA / NPP kernels** under [`lite/trt/kernel/`](https://github.com/xlite-dev/lite.ai.toolkit/tree/main/lite/trt/kernel), with `DeviceFrame`, reused buffers, and pinned + async copies.
- **Measured, not claimed.** A header-only profiler ([`lite/bench/profiler.h`](https://github.com/xlite-dev/lite.ai.toolkit/blob/main/lite/bench/profiler.h)) gives CPU-chrono + GPU-cudaEvent timings (p50 / p99 / FPS / CSV). Every optimization ships with a before/after `lite_*_bench` binary.
- **Video-shaped API.** `prepare_source()` caches the fixed source face embedding once; `process()` is the per-frame target path. The old one-shot `detect()` API remains for images and demos.
- **Multi-threaded TRT path.** `_mt` pipelines (e.g. `trt_face_restoration_mt`) run a thread pool with one
  `IExecutionContext` + `cudaStream_t` + buffer set per thread and an async task queue.

## Build 👇👇

TensorRT is the maintained backend. It needs **TensorRT 10.x** and **CUDA 12.x** (Linux only). The first
build downloads third-party libs into `third_party/` automatically.

```shell
git clone --depth=1 https://github.com/xlite-dev/lite.ai.toolkit.git
cd lite.ai.toolkit
bash ./build.sh tensorrt   # GPU / TensorRT backend
```

See [tensorrt-linux-x86_64.zh.md](./docs/tensorrt/tensorrt-linux-x86_64.zh.md) for the TensorRT/CUDA setup.

## Quick Start 🌟🌟
<div id="lite.ai.toolkit-Quick-Start"></div>

#### Flagship: FaceFusion face-swap pipeline on the GPU
End-to-end source→target face swap, fully on TensorRT. **Out of the box**, build with
`bash ./build.sh tensorrt` and run the CLI on your own images — no source editing:

```bash
# build the 5 engines once, then run:
bash ./build_facefusion_engines.sh <onnx_dir> <engine_dir>
./build/install/bin/lite_facefusion_cli <engine_dir> source.jpg target.jpg output.jpg
```

Full walkthrough: **[docs/facefusion_quickstart.md](./docs/facefusion_quickstart.md)**. The C++ API
(see [`test_lite_facefusion_pipeline.cpp`](https://github.com/xlite-dev/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_facefusion_pipeline.cpp)):

```c++
#include "lite/lite.h"
// build the 5 engines once, e.g. trtexec --onnx=gfpgan_1.4.onnx --saveEngine=gfpgan_1.4_fp32.engine
auto pipeline = lite::trt::cv::face::swap::FaceFusionPipeLine(
    face_detect_engine,        // yoloface_8n
    face_landmarks_68_engine,  // 2dfan4
    face_recognizer_engine,    // arcface_w600k_r50
    face_swap_engine,          // inswapper_128
    face_restoration_engine);  // gfpgan_1.4
// Video/server path: prepare the fixed source face once.
cv::Mat source = cv::imread(source_image_path);
pipeline.prepare_source(source, 0);

// Per target frame: process() reuses the cached source embedding.
cv::Mat target = cv::imread(target_image_path);
cv::Mat result = pipeline.process(target, 0);
cv::imwrite(save_image_path, result);

// One-shot image convenience is still available:
// pipeline.detect(source_image_path, 0, target_image_path, 0, save_image_path);
```

## Architecture 🧩

```
lite/
├── trt/          # TensorRT backend — the maintained high-performance path
│   ├── core/     # trt_handler base (engine load, buffers, streams)
│   ├── cv/       # one .h/.cpp per model + the facefusion pipeline (+ _mt variants)
│   ├── kernel/   # hand-written fused CUDA kernels (.cu/.cuh) + host-side managers
│   └── sd/       # Stable Diffusion components (clip / unet / vae / scheduler)
├── ort/          # ONNXRuntime backend — numerical reference + test host
├── bench/        # header-only profiler (CPU chrono + GPU cudaEvent, p50/p99/FPS/CSV)
└── lite.h        # single public include
```

`lite::cv` is a compile-time namespace alias resolved in [`lite/models.h`](https://github.com/xlite-dev/lite.ai.toolkit/blob/main/lite/models.h). Pin a backend explicitly with `lite::trt::cv::...` (GPU) or `lite::onnxruntime::cv::...` (CPU reference).

## ©️License
GNU General Public License v3.0

## Star History

<a href="https://www.star-history.com/?repos=star-history%2Fstar-history%2Cxlite-dev%2Flite.ai.toolkit&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=star-history/star-history%2Cxlite-dev/lite.ai.toolkit&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=star-history/star-history%2Cxlite-dev/lite.ai.toolkit&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=star-history/star-history%2Cxlite-dev/lite.ai.toolkit&type=date&legend=top-left" />
 </picture>
</a>
