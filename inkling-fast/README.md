# inkling-fast: Inkling-2-Mini on a 12 GB GPU

Branch `inkling-fast2` of llama.cpp, tuned to run [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
fully on one 12 GB NVIDIA GPU with 64k context. The model file is on Hugging Face:
[nahcmon/Inkling-2-Mini-fast-GGUF](https://huggingface.co/nahcmon/Inkling-2-Mini-fast-GGUF).

## Results (RTX 3080 Ti 12 GB, Ryzen 5 5600X, Windows 11)

Both runs: 64k context, q4_0 KV cache, same GPU, same benchmark script.

| | original (unsloth UD-Q4_K_M, stock llama.cpp) | inkling-fast |
|---|---:|---:|
| decode, short prompt | 12.2 tok/s (27.4 hand-tuned) | **76.7 tok/s** |
| decode after a 48k-token prompt | 12.5 tok/s | **56.0 tok/s** |
| prefill of a 48k-token prompt | 199 tok/s | **1632 tok/s** |
| quality, mean KLD vs reference logits | 0.0498 | 0.0498 |
| VRAM | 10.6 GB (rest in system RAM) | 11.9 GB, all on the GPU |

"Hand-tuned" is the original model with `-ngl 99 --fit off --n-cpu-moe 9`, the fastest setting that still fits;
out of the box llama.cpp keeps more experts on the CPU.

`llama-bench` (folded model, `-ub 512`): pp512 2330 tok/s, tg128 80 tok/s; at 16k depth pp512 1890 tok/s, tg128 70 tok/s.

## How the speedup was made

The original model is 15.5 GB, so on a 12 GB card part of the experts run on the CPU, and every token waits for system
RAM. Most of the gain comes from making it fit; the rest from cutting the number of GPU kernels per token.

1. **A quant that fits in VRAM.** Expert gate/up in IQ2_XS (with an importance matrix), expert down kept at the
   original NVFP4, attention in Q6_K: 10.1 GiB. With a 64k q4_0 KV cache everything stays on the GPU. Quality is the same
   as UD-Q4_K_M (KLD 0.0498 both).
2. **Folded weights** (`inkling2.fast_fold`): projections that run on the same input are packed into one tensor, so
   one matmul replaces several. Same math, fewer kernel launches.
3. **Fused depth mix** (`GGML_OP_DEPTH_MIX`, `ggml/src/ggml-cuda/depthmix.cu`): each block output is normalized once
   into a shared bank, then one kernel per mix does scores, softmax and the weighted sum.
4. **Banded flash attention** for the relative-position bias (`ggml/src/ggml-cuda/fattn-banded.cu`): the bias is
   computed inside the attention kernel instead of as a dense n_kv x n_tokens x n_head tensor. Works with quantized
   (q4_0, Hadamard-rotated) KV. This is what keeps prefill fast at 48k+.
5. **One shared attention mask per graph** instead of a copy per global-attention layer: halves the compute buffer, so
   512-token batches fit at 64k context.
6. **Fused glue kernels** (`ggml/src/ggml-cuda/inkling2.cu`, `inkling.cu`): the SiTU activation, the short
   convolution with its state update and residual add, the decode-time dynamic K/V convolution and the row L2 clip each
   run as one kernel instead of 6-11 small ops. +6.5% decode, +10% prefill, bit-exact.
7. **Server settings:** `-lm none` loads the model straight into VRAM (no 10 GB memory-mapped copy in RAM),
   `--cache-ram 0` turns off the RAM prompt cache, q4_0 KV cache, `-ub 512`.

Every engine change was checked against reference logits (KLD, top-1 agreement); none of them changed the output
beyond float rounding.

## Build (Windows, CUDA 13, VS 2022 Build Tools)

```
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 ^
      -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build-cuda --target llama-server llama-perplexity llama-bench -j 6
```

Set `CMAKE_CUDA_ARCHITECTURES` to your GPU (86 = RTX 30xx, 89 = RTX 40xx, 120 = RTX 50xx).

## Run

Put the GGUF next to the scripts (or pass `-ModelDir`), then:

```
.\inkling-fast\run-inkling.ps1           # server + web UI at http://127.0.0.1:8080, background
.\inkling-fast\stop-inkling.ps1
```

Options: `-Ctx 32768`, `-Foreground`, `-Port`, `-CacheRam 4096` (keep older chats in RAM), `-Engine <llama-server.exe>`.

Debug switches: `INKLING2_NO_FORK_FUSE=1` turns the glue fusions off, `INKLING2_FUSE_STATS=1` logs which ones fire.
