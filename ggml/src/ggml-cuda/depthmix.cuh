#include "common.cuh"

void ggml_cuda_op_depth_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_depth_mix_supported(const ggml_tensor * op);
