// [TAG_GPTQ_KT] append to ik_llama.cpp ggml/src/iqk/iqk_quantize.cpp (used by gptq_kt.py through ggml.dll)
// [TAG_GPTQ_KT] IQ2_KT encode/decode of single 256-column super-blocks with a frozen per-row scale, so a GPTQ loop
// (shrink/gptq_kt.py) can quantize slice by slice and re-aim the remaining columns. drow = the stored row float.
extern "C" GGML_API void ik_iq2kt_encode_fixed(const float * x, const float * imatrix, const float * drow, int64_t nrows, void * vy) {
    using Q = QuantizerIQ2KT;
    const auto & quantizer = iq2kt_quantizer();
    const float * vals = quantizer.values();
    block_iq2_kt * y = (block_iq2_kt *)vy;
    float w[Q::kSuperBlockSize];
    int best_idx[2*Q::kNg], keep[Q::kNg];
    for (int64_t r = 0; r < nrows; ++r) {
        const float * xr = x + r*Q::kSuperBlockSize;
        Q::set_weights(2.0f, 1, xr, imatrix, w, Q::kSuperBlockSize);
        memset(&y[r], 0, sizeof(block_iq2_kt));
        auto qs = (uint16_t *)y[r].ql;
        const float d = drow[r];
        float amax_row = 0;
        for (int j = 0; j < Q::kSuperBlockSize; ++j) amax_row = std::max(amax_row, std::abs(xr[j]));
        for (int ib = 0; ib < Q::kNblock; ++ib) {
            const float * xb = xr + ib*Q::kBlockSize;
            const float * wb = w + ib*Q::kBlockSize;
            float amax = 0;
            for (int j = 0; j < Q::kBlockSize; ++j) amax = std::max(amax, std::abs(xb[j]));
            int l0 = best_index_iq4nl(iq4k_values, 0.f), lbest = l0;
            for (int ig = 0; ig < Q::kNg; ++ig) keep[ig] = 0;
            if (d != 0 && amax > 1e-16f) {
                float scale_0 = std::max(90.f, 124.f*amax/amax_row);
                quantizer.find_best_match( amax/scale_0, xb, wb, best_idx);
                auto [dp, sp] = quantizer.find_best_scale(xb, wb, best_idx);
                quantizer.find_best_match(-amax/scale_0, xb, wb, best_idx + Q::kNg);
                auto [dm, sm] = quantizer.find_best_scale(xb, wb, best_idx + Q::kNg);
                l0 = best_index_iq4nl(iq4k_values, (sp > sm ? dp : dm)/d);
                float best_err = INFINITY;
                for (int l = std::max(0, l0 - 1); l <= std::min(15, l0 + 1); ++l) {
                    const float dl = d*iq4k_values[l];
                    quantizer.find_best_match(dl, xb, wb, best_idx);
                    float err = 0;
                    for (int ig = 0; ig < Q::kNg; ++ig) {
                        const float * q = vals + best_idx[ig]*Q::kGroupSize;
                        for (int k = 0; k < Q::kGroupSize; ++k) {
                            const float e = xb[ig*Q::kGroupSize + k] - dl*q[k];
                            err += wb[ig*Q::kGroupSize + k]*e*e;
                        }
                    }
                    if (err < best_err) {
                        best_err = err; lbest = l;
                        for (int ig = 0; ig < Q::kNg; ++ig) keep[ig] = best_idx[ig];
                    }
                }
            }
            if (ib < Q::kNblock/2) y[r].scales[ib] |= lbest;
            else                   y[r].scales[ib - Q::kNblock/2] |= lbest << 4;
            for (int ig = 0; ig < Q::kNg; ++ig) qs[ib*Q::kNg + ig] = keep[ig];
        }
    }
}

extern "C" GGML_API void ik_iq2kt_decode_fixed(const void * vx, const float * drow, int64_t nrows, float * out) {
    using Q = QuantizerIQ2KT;
    const auto & deq = iq2kt_quantizer();
    const block_iq2_kt * x = (const block_iq2_kt *)vx;
    for (int64_t r = 0; r < nrows; ++r) {
        const uint16_t * qs = (const uint16_t *)x[r].ql;
        float * yr = out + r*Q::kSuperBlockSize;
        for (int ib = 0; ib < Q::kNblock; ++ib) {
            const int nib = ib < Q::kNblock/2 ? (x[r].scales[ib] & 0xf) : (x[r].scales[ib - Q::kNblock/2] >> 4);
            const float s = drow[r]*Q::kScale*iq4k_values[nib];
            for (int ig = 0; ig < Q::kNg; ++ig) deq.set_values(qs[ib*Q::kNg + ig], yr + ib*Q::kBlockSize + ig*Q::kGroupSize, s);
        }
    }
}
