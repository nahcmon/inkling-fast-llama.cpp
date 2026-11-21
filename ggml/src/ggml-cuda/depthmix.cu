#include "depthmix.cuh"

// softmax attention over a short stack of depth states, see ggml_depth_mix
// one block per token: rms-norm cur into shared memory, score every state against every query, mix

#define DM_MAX_Q      4
#define DM_MAX_STATES 128

struct dm_seg {
    const char * data;
    int64_t      s1; // bytes between states
    int64_t      s2; // bytes between tokens
    int          n;
};

static __global__ void k_depth_mix(
        const char * __restrict__ cur, const int64_t cur_s1, const int64_t cur_s2, const int cur_ne1,
        const char * __restrict__ query, const int64_t q_s1,
        const dm_seg sa, const dm_seg sb,
        float * __restrict__ dst, const int64_t d_s1, const int64_t d_s2,
        const int ne0, const int nq, const float scale, const float eps) {
    extern __shared__ float smem[];
    __shared__ float w[DM_MAX_Q][DM_MAX_STATES];
    __shared__ float red[32];

    float * sq = smem;          // [nq, ne0]
    float * sc = smem + nq*ne0; // [ne0]

    const int t      = blockIdx.x;
    const int tid    = threadIdx.x;
    const int lane   = tid % WARP_SIZE;
    const int warp   = tid / WARP_SIZE;
    const int nwarps = blockDim.x / WARP_SIZE;
    const int ns     = sa.n + sb.n + 1;

    const float * cr = (const float *) (cur + (t % cur_ne1)*cur_s1 + (t / cur_ne1)*cur_s2);
    float ss = 0.0f;
    for (int i = tid; i < ne0; i += blockDim.x) {
        const float v = cr[i];
        sc[i] = v;
        ss += v*v;
    }
    for (int q = 0; q < nq; ++q) {
        const float * qr = (const float *) (query + q*q_s1);
        for (int i = tid; i < ne0; i += blockDim.x) {
            sq[q*ne0 + i] = qr[i];
        }
    }
    ss = warp_reduce_sum(ss);
    if (lane == 0) {
        red[warp] = ss;
    }
    __syncthreads();
    float tot = lane < nwarps ? red[lane] : 0.0f;
    tot = warp_reduce_sum(tot);
    const float r = rsqrtf(tot/ne0 + eps);
    for (int i = tid; i < ne0; i += blockDim.x) {
        sc[i] *= r;
    }
    __syncthreads();

    auto state = [&](int j) -> const float * {
        if (j < sa.n) {
            return (const float *) (sa.data + j*sa.s1 + t*sa.s2);
        }
        j -= sa.n;
        if (j < sb.n) {
            return (const float *) (sb.data + j*sb.s1 + t*sb.s2);
        }
        return sc;
    };

    for (int j = warp; j < ns; j += nwarps) {
        const float * s = state(j);
        float acc[DM_MAX_Q] = {0.0f};
        for (int i = lane; i < ne0; i += WARP_SIZE) {
            const float v = s[i];
#pragma unroll
            for (int q = 0; q < DM_MAX_Q; ++q) {
                if (q < nq) {
                    acc[q] += v*sq[q*ne0 + i];
                }
            }
        }
#pragma unroll
        for (int q = 0; q < DM_MAX_Q; ++q) {
            if (q < nq) {
                acc[q] = warp_reduce_sum(acc[q]);
                if (lane == 0) {
                    w[q][j] = acc[q]*scale;
                }
            }
        }
    }
    __syncthreads();

    if (warp < nq) {
        float mx = -INFINITY;
        for (int j = lane; j < ns; j += WARP_SIZE) {
            mx = fmaxf(mx, w[warp][j]);
        }
        mx = warp_reduce_max(mx);
        float sum = 0.0f;
        for (int j = lane; j < ns; j += WARP_SIZE) {
            const float e = expf(w[warp][j] - mx);
            w[warp][j] = e;
            sum += e;
        }
        sum = warp_reduce_sum(sum);
        const float inv = 1.0f/sum;
        __syncwarp();
        for (int j = lane; j < ns; j += WARP_SIZE) {
            w[warp][j] *= inv;
        }
    }
    __syncthreads();

    for (int i = tid; i < ne0; i += blockDim.x) {
        float acc[DM_MAX_Q] = {0.0f};
        for (int j = 0; j < ns; ++j) {
            const float v = state(j)[i];
#pragma unroll
            for (int q = 0; q < DM_MAX_Q; ++q) {
                if (q < nq) {
                    acc[q] += w[q][j]*v;
                }
            }
        }
#pragma unroll
        for (int q = 0; q < DM_MAX_Q; ++q) {
            if (q < nq) {
                dst[t*d_s2 + q*d_s1 + i] = acc[q];
            }
        }
    }
}

bool ggml_cuda_depth_mix_supported(const ggml_tensor * op) {
    const int64_t ne0 = op->ne[0];
    const int64_t nq  = op->ne[1];
    const int64_t ns  = (op->src[2] ? op->src[2]->ne[1] : 0) + (op->src[3] ? op->src[3]->ne[1] : 0) + 1;
    return nq <= DM_MAX_Q && ns <= DM_MAX_STATES && (nq + 1)*ne0*sizeof(float) <= 48*1024;
}

void ggml_cuda_op_depth_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * cur   = dst->src[0];
    const ggml_tensor * query = dst->src[1];
    const ggml_tensor * a     = dst->src[2];
    const ggml_tensor * b     = dst->src[3];

    float scale, eps;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&eps,   (const float *) dst->op_params + 1, sizeof(float));

    const int ne0 = dst->ne[0];
    const int nq  = dst->ne[1];
    const int nt  = dst->ne[2];

    dm_seg sa = { a ? (const char *) a->data : nullptr, a ? (int64_t) a->nb[1] : 0, a ? (int64_t) a->nb[2] : 0, a ? (int) a->ne[1] : 0 };
    dm_seg sb = { b ? (const char *) b->data : nullptr, b ? (int64_t) b->nb[1] : 0, b ? (int64_t) b->nb[2] : 0, b ? (int) b->ne[1] : 0 };

    // few tokens: wide blocks for memory-level parallelism on a single SM
    const int    nth  = nt < 128 ? 1024 : 256;
    const size_t smem = (size_t) (nq + 1)*ne0*sizeof(float);

    k_depth_mix<<<nt, nth, smem, ctx.stream()>>>(
        (const char *) cur->data, cur->nb[1], cur->nb[2], (int) cur->ne[1],
        (const char *) query->data, query->nb[1],
        sa, sb,
        (float *) dst->data, dst->nb[1]/sizeof(float), dst->nb[2]/sizeof(float),
        ne0, nq, scale, eps);
}
