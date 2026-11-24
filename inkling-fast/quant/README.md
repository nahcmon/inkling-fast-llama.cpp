# Quantizer: per-layer bit placement + GPTQ-style error compensation

Tools used to build `Inkling-2-Mini-fast-9.2-gptq.gguf` (9.24 GB). Every candidate was gated on `kld.py compare`
against Q8 reference logits (80 x 512 tokens of eval text); proxy losses on the calibration data were not predictive
and were never used to pick a winner.

## Results (kld.py, same 9.24 GB recipe, only the expert gate/up compensation changes)

| expert gate/up | mean KLD | top-1 |
|---|---:|---:|
| stock llama-quantize (imatrix) | 0.0491 | |
| GPTQ, damp 0.01 | 0.0482 | |
| GPTQ, damp 0.1 | 0.0469 | |
| GPTQ, damp 0.3 | 0.0448 | |
| **GPTQ, damp 1.0** | **0.0428** | 92.92% |
| GPTQ, damp 3.0 | 0.0435 | 92.73% |
| GPTQ, damp 10 | 0.0445 | 92.68% |

Same bits, 12.7% lower KLD; same speed (bench.py 83.0 vs 83.6 tok/s) and VRAM. With GPTQ a 8.98 GB recipe reaches
0.0474, better than the uncompensated 9.24 GB file.

Tried and rejected (KLD noise or worse): GPTQ on expert down (all layers 0.0434, layers 0-5 hurt), per-layer-group
damping, per-expert covariance shrunk toward the pooled one, re-solving each block's fp16 scale with frozen codes
(proxy -4.5%, KLD +12%: overfits the calibration set), output.weight in IQ4_XS (+0.0004 for -14.5 MB).

## Method (`gptq_build.py`)

Per expert e: `H_e = S_e (C + damp I) S_e`, where C is the layer's input correlation (covariance pooled over all
experts, normalized to unit diagonal) and `S_e = sqrt(diag imatrix_e)`. Columns are processed in 256-wide
super-blocks: quantize the block with the stock ggml quantizer (imatrix weights, so the codes are exactly what
llama-quantize would pick for that target), then re-aim the remaining columns to cancel the error so far:
`T_R = W_R - ((Q_F - W_F) * s_F) @ K / s_R`, `K = C_FR (C_RR)^-1`. `--plain` reproduces llama-quantize byte for byte
(`--verify pool.gguf` checks it).

## Inputs

- Source weights: `Inkling-2-Mini-UD-Q8_K_XL.gguf` (gate/up); `down_f16.gguf` from `down_pad.py prep` (down).
- Importance matrix: `imatrix_q8.gguf` (and `imatrix_q8_pad.gguf` from `down_pad.py imat`).
- Layer input covariance `covg/<tensor>.cov`: run `llama-imatrix` from this branch with `IMAT_COV_DIR=covg`
  (float32 n x n per expert tensor, summed over routed tokens). `IMAT_COVX=covx` dumps per-expert covariances.
- Pool files (`p8/*.gguf`): whole-model `llama-quantize --tensor-type-file` outputs, one per type mix; recipes take
  tensors from them by regex.

## Build

```
python gptq_build.py --tensor gate_up --type IQ2_XXS --layers 0-5   --damp 1.0 --out gptq_d1.0
python gptq_build.py --tensor gate_up --type IQ2_S   --layers 6-17  --damp 1.0 --out gptq_d1.0
python gptq_build.py --tensor gate_up --type IQ3_XXS --layers 18-23 --damp 1.0 --out gptq_d1.0
eval "args=($(cut -f2- recipe-9.2-gptq.txt))"
python assemble.py Inkling-2-Mini-fast-small.gguf Inkling-2-Mini-fast-9.2-gptq.gguf "${args[@]}"
./cand.sh <name> <assemble args...>     # assemble + kld.py compare, appends to kld.tsv
./p11_round.sh plan.txt                 # one candidate per line: name<TAB>assemble args
```

~1 min per layer on a Ryzen 5 5600X (6 workers). `LLAMA_BIN` points the scripts at another build.
