#!/usr/bin/env bash
# Build the 5 TensorRT engines the FaceFusion pipeline needs, from their ONNX files.
#
# Usage:
#   bash ./build_facefusion_engines.sh <onnx_dir> <engine_dir>
#
# <onnx_dir>   directory holding the 5 ONNX models (see docs/facefusion_quickstart.md)
# <engine_dir> where the .engine files are written (create if missing)
#
# Requires `trtexec` on PATH (ships with TensorRT 10.x). Override with TRTEXEC=...
set -euo pipefail

ONNX_DIR="${1:?usage: $0 <onnx_dir> <engine_dir>}"
ENGINE_DIR="${2:?usage: $0 <onnx_dir> <engine_dir>}"
TRTEXEC="${TRTEXEC:-trtexec}"
PYTHON="${PYTHON:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$ENGINE_DIR"

build() {
  local onnx="$ONNX_DIR/$1" engine="$ENGINE_DIR/$2"; shift 2
  if [[ ! -f "$onnx" ]]; then
    echo "[build_facefusion_engines] MISSING onnx: $onnx" >&2; exit 1
  fi
  if [[ -f "$engine" ]]; then
    echo "[build_facefusion_engines] skip (exists): $engine"; return
  fi
  echo "[build_facefusion_engines] $onnx -> $engine  ($*)"
  "$TRTEXEC" --onnx="$onnx" --saveEngine="$engine" "$@"
}

build yoloface_8n.onnx          yoloface_8n_fp16.engine          --fp16
build 2dfan4.onnx               2dfan4_fp16.engine               --fp16
build arcface_w600k_r50.onnx    arcface_w600k_r50_fp16.engine    --fp16
build inswapper_128.onnx        inswapper_128_fp16.engine        --fp16

# GFPGAN: a naive --fp16 engine blows up its StyleGAN modulated convs (grey-block / a grey
# halo around the pasted-back face). The fix is mixed precision — FP16 everywhere except the
# style_conv/to_rgb layers, which stay FP32 (build_gfpgan_fp16_engine.py). That is numerically
# identical to the FP32 engine (PSNR ~58 dB) while cutting the restoration stage ~3 ms.
# Needs the TensorRT 10.x python wheel on $PYTHON; set GFPGAN_FP32=1 to fall back to plain FP32.
GFPGAN_ENGINE="$ENGINE_DIR/gfpgan_1.4_mixed.engine"
if [[ "${GFPGAN_FP32:-0}" == "1" ]]; then
  build gfpgan_1.4.onnx         gfpgan_1.4_fp32.engine
elif [[ -f "$GFPGAN_ENGINE" ]]; then
  echo "[build_facefusion_engines] skip (exists): $GFPGAN_ENGINE"
else
  echo "[build_facefusion_engines] gfpgan_1.4.onnx -> $GFPGAN_ENGINE  (mixed fp16, style layers fp32)"
  "$PYTHON" "$HERE/build_gfpgan_fp16_engine.py" "$ONNX_DIR/gfpgan_1.4.onnx" "$GFPGAN_ENGINE"
fi

echo "[build_facefusion_engines] done -> $ENGINE_DIR"
