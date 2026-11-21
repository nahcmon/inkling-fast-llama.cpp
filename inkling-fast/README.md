# inkling-fast: Inkling-2-Mini on a 12 GB GPU

Branch `inkling-fast2` of llama.cpp, tuned to run [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
fully on one 12 GB NVIDIA GPU with 64k context. The model files are on Hugging Face:
[nahcmon/Inkling-2-Mini-fast-GGUF](https://huggingface.co/nahcmon/Inkling-2-Mini-fast-GGUF).

## Results (RTX 3080 Ti 12 GB, Ryzen 5 5600X, Windows 11)

`Inkling-2-Mini-fast-folded.gguf`, 64k context, q4_0 KV cache, `-ub 512`:

| | |
|---|---|
| decode, short context | 73-76 tok/s |
| decode at 16k depth / after a 48k prompt | 64 / 55 tok/s |
| prefill pp512 at depth 0 / 16k | 2170 / 1500 tok/s |
| prefill of a 48-60k-token prompt | ~1550 tok/s |
| VRAM / system RAM of the server | 11.9 of 12.3 GB / ~1.5 GB |
| quality, mean KLD vs reference logits | 0.0498, the same as unsloth's UD-Q4_K_M |

## What is changed

- **Banded flash attention** for the relative-position bias (`ggml/src/ggml-cuda/fattn-banded.cu`): no dense
  n_kv x n_tokens x n_head bias tensor in prefill; works with quantized (q4_0, Hadamard-rotated) KV.
- **Fused depth mix** (`GGML_OP_DEPTH_MIX`, `ggml/src/ggml-cuda/depthmix.cu`): each block output is normalized once
  into a shared bank, then one kernel per mix does scores, softmax and the weighted sum.
- **Folded weights** (`inkling2.fast_fold`): projections repacked so fewer kernels run per token (same math).
- **Fused row L2 clip** (`ggml/src/ggml-cuda/inkling.cu`).
- **One shared attention mask per graph** instead of a copy per global-attention layer: halves the compute buffer, so
  512-token batches fit at 64k context.
- The launcher loads the model straight into VRAM (`-lm none`) and disables the RAM prompt cache (`--cache-ram 0`).

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
