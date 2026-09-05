//
// Created by zhuohaoyang on 2026/8/28.
//

#include "yolov26_preprocess.cuh"

#include <cstddef>

namespace
{
  __device__ __forceinline__ int clamp_int(int value, int lower, int upper)
  {
    return max(lower, min(value, upper));
  }

  __device__ __forceinline__ float interpolate_channel(
      const unsigned char *src, std::size_t pitch,
      int src_width, int src_height, int dst_x, int dst_y,
      int resized_width, int resized_height, int channel)
  {
    // Match OpenCV's CV_8U INTER_LINEAR path: half-pixel coordinates,
    // 11-bit quantized horizontal/vertical coefficients, and the same
    // two-stage fixed-point rounding before uint8 normalization.
    float fx = static_cast<float>((static_cast<double>(dst_x) + 0.5) *
                                  static_cast<double>(src_width) /
                                  static_cast<double>(resized_width) - 0.5);
    int sx = static_cast<int>(floorf(fx));
    fx -= static_cast<float>(sx);
    if (sx < 0)
    {
      sx = 0;
      fx = 0.0f;
    }
    if (sx >= src_width - 1)
    {
      sx = src_width - 1;
      fx = 0.0f;
    }

    float fy = static_cast<float>((static_cast<double>(dst_y) + 0.5) *
                                  static_cast<double>(src_height) /
                                  static_cast<double>(resized_height) - 0.5);
    const int sy = static_cast<int>(floorf(fy));
    fy -= static_cast<float>(sy);

    const int x0 = sx;
    const int x1 = clamp_int(sx + 1, 0, src_width - 1);
    const int y0 = clamp_int(sy, 0, src_height - 1);
    const int y1 = clamp_int(sy + 1, 0, src_height - 1);
    constexpr int coefficient_scale = 1 << 11;
    const int alpha0 = __float2int_rn((1.0f - fx) * coefficient_scale);
    const int alpha1 = __float2int_rn(fx * coefficient_scale);
    const int beta0 = __float2int_rn((1.0f - fy) * coefficient_scale);
    const int beta1 = __float2int_rn(fy * coefficient_scale);

    const unsigned char *row0 = src + static_cast<std::size_t>(y0) * pitch;
    const unsigned char *row1 = src + static_cast<std::size_t>(y1) * pitch;
    const int horizontal0 = static_cast<int>(row0[x0 * 3 + channel]) * alpha0 +
                            static_cast<int>(row0[x1 * 3 + channel]) * alpha1;
    const int horizontal1 = static_cast<int>(row1[x0 * 3 + channel]) * alpha0 +
                            static_cast<int>(row1[x1 * 3 + channel]) * alpha1;
    const int value = ((((beta0 * (horizontal0 >> 4)) >> 16) +
                        ((beta1 * (horizontal1 >> 4)) >> 16) + 2) >> 2);
    return static_cast<float>(clamp_int(value, 0, 255));
  }

  __global__ void yolov26_preprocess_kernel(
      const unsigned char *src, std::size_t src_pitch,
      int src_width, int src_height, float *dst,
      int dst_width, int dst_height,
      int resized_width, int resized_height, int left, int top)
  {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= dst_width || y >= dst_height) return;

    const int pixel = y * dst_width + x;
    const int plane = dst_width * dst_height;
    const int resized_x = x - left;
    const int resized_y = y - top;
    constexpr float inverse_255 = 1.0f / 255.0f;

    if (resized_x < 0 || resized_x >= resized_width ||
        resized_y < 0 || resized_y >= resized_height)
    {
      const float padding = 114.0f * inverse_255;
      dst[pixel] = padding;
      dst[plane + pixel] = padding;
      dst[2 * plane + pixel] = padding;
      return;
    }

    dst[pixel] = interpolate_channel(src, src_pitch, src_width, src_height,
                                     resized_x, resized_y,
                                     resized_width, resized_height, 2) * inverse_255;
    dst[plane + pixel] = interpolate_channel(src, src_pitch, src_width, src_height,
                                             resized_x, resized_y,
                                             resized_width, resized_height, 1) * inverse_255;
    dst[2 * plane + pixel] = interpolate_channel(src, src_pitch, src_width, src_height,
                                                 resized_x, resized_y,
                                                 resized_width, resized_height, 0) * inverse_255;
  }
}

cudaError_t trtcv::kernel::launch_yolov26_preprocess(
    const unsigned char *src_bgr, std::size_t src_pitch,
    int src_width, int src_height, float *dst_chw,
    int dst_width, int dst_height,
    int resized_width, int resized_height, int left, int top,
    cudaStream_t stream)
{
  if (!src_bgr || !dst_chw || src_width <= 0 || src_height <= 0 ||
      dst_width <= 0 || dst_height <= 0 || resized_width <= 0 || resized_height <= 0)
    return cudaErrorInvalidValue;

  const dim3 block(16, 16);
  const dim3 grid((dst_width + block.x - 1) / block.x,
                  (dst_height + block.y - 1) / block.y);
  yolov26_preprocess_kernel<<<grid, block, 0, stream>>>(
      src_bgr, src_pitch, src_width, src_height, dst_chw,
      dst_width, dst_height, resized_width, resized_height, left, top);
  return cudaGetLastError();
}
