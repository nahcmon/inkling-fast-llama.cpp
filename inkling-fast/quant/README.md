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

### Round 2: `Inkling-2-Mini-fast-9.3-tuned.gguf` (same 9.24 GB, same tensor types and speed)

| step | mean KLD | KLD99 | top-1 |
|---|---:|---:|---:|
| 9.2-gptq (above) | 0.04282 | 0.4240 | 92.92% |
| + MoE block-output scale tuning only (`moe_recon.py`) | 0.04106 | 0.4065 | 92.84% |
| GPTQ rebuilt with a 3x larger chat + docs calibration only | 0.04100 | 0.3924 | 92.76% |
| **both (scale tuning on top of the new GPTQ file)** | **0.03909** | **0.3703** | **93.10%** |

Calibration: `cmp/calib2.txt` (1.4 MB, chat transcripts + the old docs text), imatrix run with `IMAT_DUMP_DIR`
(`IMAT_DUMP_EVERY=8`), layer covariance from that dump (`cov_from_dump.py`), `gptq_build.py --imatrix imatrix2.gguf
--cov covg2 --damp 1.0`, then `moe_recon.py --cand <that file> --layers 0-23 --epochs 8` (about 50 min on the 3080 Ti):
per layer the fp16 block scales of expert gate/up and down get learned log-multipliers (codes frozen) so the MoE
block output, routing and activation included, matches the Q8 reference on the dumped inputs; held-out MoE error
drops 7-19% per layer. llama-bench: 89.0 vs 89.2 tok/s decode, 2459 vs 2490 tok/s pp512.

Tried and rejected (KLD noise or worse): GPTQ on expert down (all layers 0.0434, layers 0-5 hurt), per-layer-group
damping, per-expert covariance shrunk toward the pooled one, re-solving each block's fp16 scale with frozen codes
(proxy -4.5%, KLD +12%: overfits the calibration set), output.weight in IQ4_XS (+0.0004 for -14.5 MB).

Also rejected (kld.py, layers 6-17 expert gate/up swapped, all through the same Q8_0 round trip, control 0.0429):
- In-block GPTQ (`gptq_vq.py`: compensation inside each 256 block, group by group): 0.0441, despite a 9-12% lower proxy.
- ik_llama.cpp trellis IQ2_KT (2.125 bpw, `kt_test.py`): 0.0546.
- GPTQ on top of IQ2_KT (`gptq_kt.py` + `ik_gptq_kt.cpp`: row scale frozen, one super-block at a time): 0.0515. Saves
  about 380 MB on those layers but closes only a quarter of the gap, so no CUDA port of the trellis decode.

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

- `realtest.py` measures decode, prefill, and GPU shared memory in a five-turn vision chat. Use `--model-dir` to point it at the deployed model folder.
- `vspeed_long.py` checks uncached long prompts and 128-token decode. Use `--model-dir` for the vision projector, `--corpus` for the prompt text, and `--ub 512 --kv-global q4_0` for Long.
