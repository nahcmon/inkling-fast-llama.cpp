#include "inkling.cuh"

static __global__ void inkling_clip_kernel(
        const float * __restrict__ x,
        const float * __restrict__ s,
        float       * __restrict__ dst,
        const int ne0, const int ne1, const int nrows, const int s_ne1,
        const float thr, const float sb_s, const float sb_b, const float cmin, const float cmax) {
    const int row = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    if (row >= nrows) {
        return;
    }
    const int lane = threadIdx.x % WARP_SIZE;

    const float * xr = x   + (int64_t) row * ne0;
    float       * dr = dst + (int64_t) row * ne0;
    const float * sr = s ? s + (int64_t) ((row % ne1) % s_ne1) * ne0 : nullptr;

    float ss = 0.0f;
    for (int i = lane; i < ne0; i += WARP_SIZE) {
        const float v = sr ? xr[i] * sr[i] : xr[i];
        ss += v * v;
    }
    ss = warp_reduce_sum(ss);

    const float norm = sqrtf(ss) * sb_s + sb_b;
    const float f    = fminf(fmaxf(thr / norm, cmin), cmax);

    for (int i = lane; i < ne0; i += WARP_SIZE) {
        const float v = sr ? xr[i] * sr[i] : xr[i];
        dr[i] = v * f;
    }
}

void ggml_cuda_op_inkling_clip(ggml_backend_cuda_context & ctx,
                               const ggml_tensor * x,
                               const ggml_tensor * scale,
                               float thr, float sb_s, float sb_b, float cmin, float cmax,
                               ggml_tensor * dst) {
    const int ne0   = (int) x->ne[0];
    const int ne1   = (int) x->ne[1];
    const int nrows = (int) ggml_nrows(x);

    const int rows_per_block = 4;
    const dim3 block(WARP_SIZE * rows_per_block, 1, 1);
    const dim3 grid((nrows + rows_per_block - 1) / rows_per_block, 1, 1);

    inkling_clip_kernel<<<grid, block, 0, ctx.stream()>>>(
        (const float *) x->data, scale ? (const float *) scale->data : nullptr, (float *) dst->data,
        ne0, ne1, nrows, scale ? (int) scale->ne[1] : 1, thr, sb_s, sb_b, cmin, cmax);
}
