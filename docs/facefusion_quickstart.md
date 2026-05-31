# FaceFusion Pipeline — Quickstart (TensorRT)

Run the flagship end-to-end face-swap pipeline (detect → 68 landmarks → recognize →
swap → restore) on your own images, on an NVIDIA GPU. Linux only, **TensorRT 10.x +
CUDA 12.x**.

## 1. Build

```bash
git clone --depth=1 https://github.com/xlite-dev/lite.ai.toolkit.git
cd lite.ai.toolkit
bash ./build.sh tensorrt
```

Binaries land in `build/install/bin/` (the CLI runner is `lite_facefusion_cli`).

## 2. Get the 5 ONNX models

The pipeline uses these 5 models (the standard FaceFusion / InsightFace assets):

| Stage | ONNX file |
|--|--|
| face detect | `yoloface_8n.onnx` |
| 68 landmarks | `2dfan4.onnx` |
| face recognize | `arcface_w600k_r50.onnx` |
| face swap | `inswapper_128.onnx` |
| face restore | `gfpgan_1.4.onnx` |

Put all 5 in one directory, e.g. `~/ff_onnx/`.

## 3. Build the TensorRT engines

```bash
bash ./build_facefusion_engines.sh ~/ff_onnx ~/ff_engines
```

This runs `trtexec` once per model and writes the 5 `.engine` files into `~/ff_engines/`.
GFPGAN is kept FP32 on purpose (FP16 produces grey-block artifacts on its StyleGAN
modulated convs); the other four are FP16. Engines are GPU/TensorRT-version specific —
rebuild them if you change GPU or TensorRT version.

## 4. Run

```bash
./build/install/bin/lite_facefusion_cli \
    ~/ff_engines \
    source.jpg \          # face to take
    target.jpg \          # image to paste it onto
    output.jpg            # result

# optionally pick which detected face to use on each side (default 0 0):
#   ... output.jpg <src_face_idx> <tgt_face_idx>
```

That's it — `output.jpg` is the swapped + restored result.

## Performance

The face-restoration stage is GPU-fused (paste-back + preprocess moved into CUDA
kernels): **78.2 ms → 17.7 ms (4.4×)** on an RTX 4090. See the
[Benchmark](../README.md#benchmark) section. Other stages are being optimized stage
by stage.
