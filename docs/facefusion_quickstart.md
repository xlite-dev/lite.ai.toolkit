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

This runs `trtexec` for four of the models (FP16) and writes the `.engine` files into
`~/ff_engines/`. GFPGAN is built as a **mixed-precision** engine via
`build_gfpgan_fp16_engine.py`: a naive `--fp16` GFPGAN blows up its StyleGAN modulated
convs (a grey halo around the pasted-back face), so the style_conv/to_rgb layers are kept
FP32 and the rest run FP16. That is numerically identical to the FP32 engine (PSNR ~58 dB)
while cutting the restoration stage ~3 ms (≈28.6 → 31 FPS on an RTX 4090).

The mixed build needs the **TensorRT 10.x python wheel** on `python3` (ships in the TRT
tarball under `python/`, e.g. `pip install /usr/local/tensorrt/python/tensorrt-10.*-cp3*-*.whl`).
If you can't set that up, run `GFPGAN_FP32=1 bash ./build_facefusion_engines.sh ...` to fall
back to a plain FP32 GFPGAN engine. Engines are GPU/TensorRT-version specific — rebuild them
if you change GPU or TensorRT version.

## 4. Run

```bash
./build/install/bin/lite_facefusion_cli \
    ~/ff_engines \
    source.jpg \
    target.jpg \
    output.jpg

# optionally pick which detected face to use on each side (default 0 0):
#   ... output.jpg <src_face_idx> <tgt_face_idx>
```

That's it — `output.jpg` is the swapped + restored result. `source.jpg` is the face to take,
and `target.jpg` is the image to paste it onto.

## Performance

The benchmark path is video-shaped: prepare the fixed source face once, then time
per-frame `process(target)`. On an RTX 4090, FP16 deployment, the current pipeline
runs at **23.6 ms / frame (42.3 FPS)**. The pipeline now does one full-frame H2D
upload for the target and one full-frame D2H download for the final result; the
swap → restoration boundary stays GPU-resident. See the
[Benchmark](../README.md#benchmark) section.
