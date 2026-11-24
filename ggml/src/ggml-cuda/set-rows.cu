#include "set-rows.cuh"
#include "cpy-utils.cuh"

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Experiment (GGML_KV_FAKEQ="K,V", each q4|u3|n3|u2): store the global KV cache as q4_0 blocks whose codes are
// limited to a 3- or 2-bit grid, to measure the quality of a 3/2-bit cache before writing real kernels.
//   u3: uniform 8 levels -4..3, n3: nonuniform 8 levels +-1,2,4,7, u2: uniform 4 levels -2..1.
// The scale is the one (of 13 tried) with the least squared error.
template <int G>
static __device__ void quantize_f32_fakeq_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    constexpr int NL = G == 2 ? 4 : 8;
    const int L[8] = { G == 0 ? -4 : G == 1 ? -7 : -2, G == 0 ? -3 : G == 1 ? -4 : -1, G == 0 ? -2 : G == 1 ? -2 : 0,
                       G == 0 ? -1 : G == 1 ? -1 : 1,  G == 0 ? 0 : 1, G == 0 ? 1 : 2, G == 0 ? 2 : 4, G == 0 ? 3 : 7 };
    const float lneg = G == 0 ? 4.0f : G == 1 ? 7.0f : 2.0f;
    float amax = 0.0f, vmax = 0.0f;
    for (int j = 0; j < QK4_0; ++j) {
        if (amax < fabsf(x[j])) { amax = fabsf(x[j]); vmax = x[j]; }
    }
    // the largest magnitude maps to the most negative level (or its mirror for the symmetric grid)
    const float d0 = G == 1 ? amax / lneg : vmax / -lneg;
    float best_d = d0, best_e = 3.4e38f;
    for (int c = 0; c < 13; ++c) {
        const float d = d0 * (0.6f + 0.05f * c);
        if (d == 0.0f) { break; }
        const float id = 1.0f / d;
        float e = 0.0f;
        for (int j = 0; j < QK4_0; ++j) {
            const float v = x[j] * id;
            float bd = 1e30f;
            for (int k = 0; k < NL; ++k) { const float t = (v - L[k]) * (v - L[k]); bd = fminf(bd, t); }
            e += bd * d * d;
        }
        if (e < best_e) { best_e = e; best_d = d; }
    }
    y->d = best_d;
    const float id = best_d ? 1.0f / best_d : 0.0f;
    uint8_t q[QK4_0];
    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j] * id;
        int bk = 0; float bd = 1e30f;
        for (int k = 0; k < NL; ++k) { const float t = (v - L[k]) * (v - L[k]); if (t < bd) { bd = t; bk = k; } }
        q[j] = (uint8_t) (L[bk] + 8);
    }
    for (int j = 0; j < QK4_0/2; ++j) {
        y->qs[j] = q[j] | (q[QK4_0/2 + j] << 4);
    }
}

// 0 = normal q4_0, else 1 + grid (u3, n3, u2), for the global K and V caches
static int fakeq_mode(const ggml_tensor * dst) {
    static int mk = -1, mv = -1;
    if (mk < 0) {
        mk = mv = 0;
        const char * e = getenv("GGML_KV_FAKEQ");
        auto parse = [](const char * s) { return !strncmp(s, "u3", 2) ? 1 : !strncmp(s, "n3", 2) ? 2 : !strncmp(s, "u2", 2) ? 3 : 0; };
        if (e) { mk = parse(e); const char * c = strchr(e, ','); mv = c ? parse(c + 1) : mk; }
    }
    const ggml_tensor * t = dst;
    while (t->view_src) { t = t->view_src; }
    if (!strncmp(t->name, "cache_gk_", 9)) { return mk; }
    if (!strncmp(t->name, "cache_gv_", 9)) { return mv; }
    return 0;
}

// Generic quantized set_rows kernel template
template <typename src_t, typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const src_t * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const src_t * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    if constexpr (std::is_same_v<src_t, float>) {
        quantize_func(src_block, dst_block);
    } else {
        // the quantize functions take F32 input: widen the block first
        float src_block_f32[qk];
#pragma unroll
        for (int j = 0; j < qk; ++j) {
            src_block_f32[j] = ggml_cuda_cast<float>(src_block[j]);
        }
        quantize_func(src_block_f32, dst_block);
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename src_t, typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const src_t * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<src_t, idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * src0_ptr,
                                  const idx_t * src1_ptr,
                                  dst_t * dst_ptr,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const src_t * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const idx_t * GGML_CUDA_RESTRICT src1 = src1_ptr;
    dst_t       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    ggml_cuda_pdl_lc();

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_size, block_size, 0, stream);
        ggml_cuda_kernel_launch(k_set_rows<src_t, idx_t, dst_t>, launch_params,
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
            s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
            ne11_fd, ne12_fd);
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F8_E4M3) {
        set_rows_cuda(
            src0_d, src1_d, (ggml_fp8_e4m3_t *) dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q3_0) {
        set_rows_cuda_quant<src_t, idx_t, block_q3_0, QK3_0, quantize_f32_q3_0_block>(
            src0_d, src1_d, (block_q3_0 *) dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0 && fakeq_mode(dst) != 0) {
        const int m = fakeq_mode(dst);
#define FAKEQ_CALL(F) set_rows_cuda_quant<src_t, idx_t, block_q4_0, QK4_0, F>(             src0_d, src1_d, (block_q4_0*)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13,             nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3, stream)
        if (m == 1) { FAKEQ_CALL(quantize_f32_fakeq_block<0>); }
        else if (m == 2) { FAKEQ_CALL(quantize_f32_fakeq_block<1>); }
        else { FAKEQ_CALL(quantize_f32_fakeq_block<2>); }
#undef FAKEQ_CALL
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<src_t, idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<src_t, idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<src_t, idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<src_t, idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<src_t, idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<src_t, idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}

template<>
void set_rows_cuda<half, int32_t>(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const half    * src0_d = (const half *)src0->data;
    const int32_t * src1_d = (const int32_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}

template<>
void set_rows_cuda<half, int64_t>(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const half    * src0_d = (const half *)src0->data;
    const int64_t * src1_d = (const int64_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}

void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16 || (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16));
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src0->type == GGML_TYPE_F32) {
        if (src1->type == GGML_TYPE_I64) {
            set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
        } else {
            set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
        }
    } else if (src0->type == GGML_TYPE_F16) {
        if (src1->type == GGML_TYPE_I64) {
            set_rows_cuda<half, int64_t>(ctx, src0, src1, dst);
        } else {
            set_rows_cuda<half, int32_t>(ctx, src0, src1, dst);
        }
    } else if (src0->type == GGML_TYPE_BF16) {
        if (src1->type == GGML_TYPE_I64) {
            set_rows_cuda<nv_bfloat16, int64_t>(ctx, src0, src1, dst);
        } else {
            set_rows_cuda<nv_bfloat16, int32_t>(ctx, src0, src1, dst);
        }
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(src0->type));
    }
}
