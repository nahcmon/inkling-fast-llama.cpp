#include "common.cuh"

// Inkling-2 fused kernels (decode-path kernel-count reduction).

// row-wise L2 clip: dst = y * clamp(thr / (sqrt(sum(y^2))*sb_s + sb_b), cmin, cmax), y = x * scale (scale optional, broadcast over rows)
void ggml_cuda_op_inkling_clip(ggml_backend_cuda_context & ctx,
                               const ggml_tensor * x,
                               const ggml_tensor * scale,
                               float thr, float sb_s, float sb_b, float cmin, float cmax,
                               ggml_tensor * dst);
