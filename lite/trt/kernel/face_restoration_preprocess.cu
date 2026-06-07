#include "face_restoration_preprocess.cuh"

// One thread per crop pixel. Reads interleaved BGR uint8, writes planar RGB float (CHW),
// normalized as out = v*scale + bias. Channel mapping: R->plane0, G->plane1, B->plane2.
// restoration: scale=1/127.5, bias=-1 ([-1,1]); swap: scale=1/255, bias=0 ([0,1]).
__global__ void face_restoration_preprocess_kernel(const unsigned char* crop, float* out, int H, int W,
                                                   float scale, float bias) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int i = (y * W + x) * 3;
    float b = static_cast<float>(crop[i + 0]);
    float g = static_cast<float>(crop[i + 1]);
    float r = static_cast<float>(crop[i + 2]);

    int plane = H * W;
    int off = y * W + x;
    out[0 * plane + off] = r * scale + bias;
    out[1 * plane + off] = g * scale + bias;
    out[2 * plane + off] = b * scale + bias;
}
