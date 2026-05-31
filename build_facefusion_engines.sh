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

mkdir -p "$ENGINE_DIR"

# onnx_basename  engine_basename  extra_flags
# GFPGAN stays FP32 on purpose: FP16 makes its StyleGAN modulated convs blow up
# (grey-block artifacts). The other four are fine in FP16.
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
build gfpgan_1.4.onnx           gfpgan_1.4_fp32.engine

echo "[build_facefusion_engines] done -> $ENGINE_DIR"
