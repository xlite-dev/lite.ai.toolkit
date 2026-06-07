#include "paste_back.cuh"
#include <cuda_runtime.h>

__global__ void paste_back_kernel(const float* inverse_vision_frame,
                                  const float* temp_frame,
                                  const float* inverse_mask,
                                  float* output,
                                  int width,
                                  int height,
                                  int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        float inverse_weight = 1.0f - inverse_mask[y * width + x];

        for (int c = 0; c < channels; ++c) {
            output[idx + c] = inverse_mask[y * width + x] * inverse_vision_frame[idx + c] +
                              inverse_weight * temp_frame[idx + c];
        }
    }
}

// ---------------- fused inverse-mapping paste_back ----------------
// Single-channel bilinear sample; out-of-range taps read 0 (matches cv::BORDER_CONSTANT 0)
__device__ __forceinline__ float bilinear1(const float* img, int W, int H, float u, float v) {
    int x0 = floorf(u), y0 = floorf(v);
    float fx = u - x0, fy = v - y0;
    float a = (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H) ? img[y0 * W + x0] : 0.f;
    float b = (x0 + 1 >= 0 && x0 + 1 < W && y0 >= 0 && y0 < H) ? img[y0 * W + (x0 + 1)] : 0.f;
    float c = (x0 >= 0 && x0 < W && y0 + 1 >= 0 && y0 + 1 < H) ? img[(y0 + 1) * W + x0] : 0.f;
    float d = (x0 + 1 >= 0 && x0 + 1 < W && y0 + 1 >= 0 && y0 + 1 < H) ? img[(y0 + 1) * W + (x0 + 1)] : 0.f;
    return (a * (1.f - fx) + b * fx) * (1.f - fy) + (c * (1.f - fx) + d * fx) * fy;
}

// Three-channel (interleaved BGR) bilinear sample; out-of-range taps read 0
__device__ __forceinline__ float bilinear3(const float* img, int W, int H, float u, float v, int ch) {
    int x0 = floorf(u), y0 = floorf(v);
    float fx = u - x0, fy = v - y0;
    float a = (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H) ? img[(y0 * W + x0) * 3 + ch] : 0.f;
    float b = (x0 + 1 >= 0 && x0 + 1 < W && y0 >= 0 && y0 < H) ? img[(y0 * W + (x0 + 1)) * 3 + ch] : 0.f;
    float c = (x0 >= 0 && x0 < W && y0 + 1 >= 0 && y0 + 1 < H) ? img[((y0 + 1) * W + x0) * 3 + ch] : 0.f;
    float d = (x0 + 1 >= 0 && x0 + 1 < W && y0 + 1 >= 0 && y0 + 1 < H) ? img[((y0 + 1) * W + (x0 + 1)) * 3 + ch] : 0.f;
    return (a * (1.f - fx) + b * fx) * (1.f - fy) + (c * (1.f - fx) + d * fx) * fy;
}

__global__ void paste_back_fused_kernel(const unsigned char* temp,
                                        const float* crop,
                                        const float* mask,
                                        const float* M,
                                        unsigned char* out,
                                        int W, int H, int Cw, int Ch,
                                        float blend_alpha) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int oidx = (y * W + x) * 3;

    // original coords -> crop coords (use M directly, no inversion needed)
    float u = M[0] * x + M[1] * y + M[2];
    float v = M[3] * x + M[4] * y + M[5];

    // pixels outside the crop get mask=0 and just copy temp (matches CPU BORDER_CONSTANT 0)
    float m = bilinear1(mask, Cw, Ch, u, v);
    m = fminf(fmaxf(m, 0.f), 1.f);
    // fold the face-enhancer blend (result = temp*(1-a*mask) + crop*(a*mask)); a=1 -> plain paste,
    // a=0.8 -> restoration's blend_frame(target,0.2 / paste,0.8) collapsed into the mask.
    m *= blend_alpha;

    if (m > 0.f) {
        float w = 1.f - m;
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            float cs = bilinear3(crop, Cw, Ch, u, v, c);
            float ts = static_cast<float>(temp[oidx + c]);
            float o = m * cs + w * ts;
            out[oidx + c] = static_cast<unsigned char>(fminf(fmaxf(o + 0.5f, 0.f), 255.f));
        }
    } else {
        out[oidx + 0] = temp[oidx + 0];
        out[oidx + 1] = temp[oidx + 1];
        out[oidx + 2] = temp[oidx + 2];
    }
}
