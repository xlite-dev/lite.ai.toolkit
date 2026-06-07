#!/usr/bin/env python
# Build a *mixed-precision* TensorRT engine for GFPGAN that the lite.ai.toolkit C++
# (TensorRT 10.1) can load — fast FP16 everywhere EXCEPT the StyleGAN modulated convs,
# which are kept in FP32.
#
# Why this exists:
#   A naive `trtexec --fp16` GFPGAN engine produces grey-block artifacts: the StyleGAN
#   "modulated conv" demodulation (sum-of-squares -> rsqrt) overflows/underflows in FP16.
#   The clean fix is to keep just those layers (style_conv* / to_rgb*) in FP32 and run the
#   rest in FP16. On TensorRT 10.1 the "strong typing via Cast nodes in the ONNX" route
#   crashes (matchTypeSpec); the route that works is weak FP16 + OBEY_PRECISION_CONSTRAINTS
#   with per-layer FP32 precision set through the builder API (this script).
#
# Result on RTX 4090 / TRT 10.1: restoration infer 10.8 -> 8.0 ms, output numerically clean
# (no grey blocks); facefusion pipeline 36.6 -> 33.1 ms (27 -> 30 FPS).
#
# Requirements: the TensorRT 10.1 *python* wheel (ships in the TRT tarball under python/),
# e.g.  python -m venv env && env/bin/pip install /usr/local/tensorrt/python/tensorrt-10.1.0-cp312-*.whl
#
# Usage:
#   LD_LIBRARY_PATH=/usr/local/tensorrt/lib:/usr/local/cuda/lib64 \
#     python build_gfpgan_fp16_engine.py <gfpgan.onnx> <out.engine>
#
import sys
import os
import tensorrt as trt

# Substring match on layer names; these are the StyleGAN modulated convs that must stay FP32.
KEEP_FP32 = ("style_conv", "to_rgb")
FLOAT_TYPES = (trt.float32, trt.float16)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    onnx_path, engine_path = sys.argv[1], sys.argv[2]

    log = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(log)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, log)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            sys.exit(1)

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    def all_float(layer):
        return layer.num_outputs > 0 and all(
            layer.get_output(j).dtype in FLOAT_TYPES for j in range(layer.num_outputs)
        )

    pinned = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if not any(k in layer.name for k in KEEP_FP32):
            continue
        # Only float compute layers; skip Int64/shape Constants (can't be FP32-typed).
        if layer.type == trt.LayerType.CONSTANT or not all_float(layer):
            continue
        layer.precision = trt.float32
        for j in range(layer.num_outputs):
            layer.set_output_type(j, trt.float32)
        pinned += 1
    print(f"network layers={network.num_layers}  pinned to fp32={pinned}", flush=True)

    print("building serialized engine (this is slow on the first build)...", flush=True)
    serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        print("BUILD FAILED")
        sys.exit(1)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"OK wrote {engine_path} ({os.path.getsize(engine_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
