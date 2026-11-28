# inkling-fast: Inkling-2-Mini on a 12 GB GPU

Branch `inkling-fast2` of llama.cpp, tuned to run [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
fully on one 12 GB NVIDIA GPU: 128k context, image input, and 8 requests at once (chats or OpenCode subagents)
from one shared KV cache. The model files are on Hugging Face:
[nahcmon/Inkling-2-Mini-fast-GGUF](https://huggingface.co/nahcmon/Inkling-2-Mini-fast-GGUF).

Current file: `Inkling-2-Mini-Uncensored-fast.gguf` (9.24 GB), a quant of
[AtomicChat's uncensored fine-tune](https://huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF) with the same
recipe: mean KLD 0.0376 vs its Q8 reference (kld.py), same speed as the previous `fast-9.3-tuned`. It answers many
requests the original refuses; you are responsible for how you use it. Speeds are in [Run](#run) below.

## Previous release: 128k context, vision, 8 requests at once

`Inkling-2-Mini-fast-small.gguf` (9.5 GB, since replaced by 9.3-tuned) + `mmproj-Inkling-2-Mini-Q8_0.gguf`, 128k
shared context, q4_0 KV, `-np 8 -kvu -ub 256`. VRAM 11.4 GB with vision loaded. End-to-end HTTP benchmark (256 tokens per
request, all requests started together, temperature 1.0, mean of 2 rounds):

| requests at once | tok/s each request sees | total tok/s |
|---:|---:|---:|
| 1 | 81.6 | 77.9 |
| 2 | 63.4 | 117.8 |
| 4 | 42.2 | 155.4 |
| 8 | 26.7 | **196.5** |

Quality: mean KLD of fast-small vs fast-folded 0.013 (top-1 95.8%); fast-folded matches the original UD-Q4_K_M
(0.0498 both, table below). The q4_0 KV cache adds 0.007 vs an f16 cache.

## Time to first token in a running chat

A follow-up message in an existing chat (only the new turn is processed, the rest is cached) shows its first token
after (ttft.py, median of the follow-up turns, `-c 163840 -np 8 -kvu`, other chats in the shared cache):

| chat history | time to first token |
|---:|---:|
| short | 131 ms |
| 8k tokens | 161 ms |
| 42k tokens | 219 ms |

Decode speed is unchanged (82-83 tok/s for one chat, bench.py). See items 15-17 below.

## First release: 64k context, one chat

RTX 3080 Ti 12 GB, Ryzen 5 5600X, Windows 11. `Inkling-2-Mini-fast-folded.gguf` (10.1 GiB). Both runs: 64k context, q4_0 KV cache, same GPU, same benchmark script.

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

### Several chats at once

`run-inkling.ps1 -Parallel N` serves N chats from one shared 64k KV cache (`-np N -kvu`), so VRAM stays at 12.0 GB for
any N. End-to-end HTTP benchmark, 256 tokens per chat, all chats started together:

| chats | tok/s each chat sees | total tok/s | VRAM |
|---:|---:|---:|---:|
| 1 | 77.8 | 75.5 | 11.9 GB |
| 2 | 57.3 | 108.7 | 12.0 GB |
| 4 | 38.3 | 146.6 | 12.0 GB |
| 8 | 25.6 | **195.7** | 12.0 GB |

`llama-batched-bench` (decode only, no HTTP or prompt processing) reaches 124 / 174 / 237 tok/s at 2 / 4 / 8 chats
(82 / 98 / 98 before the MoE kernel fix below). For a MoE model it overstates real throughput: all its sequences
share one prompt and decode in lockstep, so they route to the same experts and load fewer expert weights per step
than independent chats do. Use an end-to-end benchmark with different prompts for real numbers.

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
7. **Faster MoE kernel for several chats** (`ggml/src/ggml-cuda/mmvq.cu`): when the per-expert scale of the NVFP4
   down projection was fused into the expert matmul, the kernel still carried the unused gate accumulators. With
   more than one token per expert that made it 50x slower (2.2 ms per call at 8 chats). It now has a gate-free
   variant: 8 chats went from 98 to 237 tok/s, one chat is unchanged, output identical (same perplexity).
8. **Server settings:** `-lm none` loads the model straight into VRAM (no 10 GB memory-mapped copy in RAM),
   `--cache-ram 0` turns off the RAM prompt cache, q4_0 KV cache, `-ub 512` (`-ub 256` at 128k). `--ctx-checkpoints 2`: Inkling keeps a
   sliding-window and recurrent state that the server snapshots (~50 MB each) to resume a chat; the default of 32
   per chat let RAM grow by ~400 MB per round with 4 long chats (1.3 to 6 GB). Two per chat is enough to resume the
   next turn without reprocessing, and RAM stays flat.
9. **A smaller quant for 128k + vision** (`Inkling-2-Mini-fast-small.gguf`, 9.5 GB): expert down NVFP4 to IQ3_XXS
   (rows padded to the IQ3 block size, importance-matrix weighted), the rest as fast-folded. Frees the VRAM for a
   128k cache and the Q8_0 vision projector. Tried and rejected: Hadamard-rotated expert down (worse at IQ3_XXS),
   IQ2_S gate/up, activation-based pruning, sharing K/V between the looped passes (KLD 0.02-0.06).
10. **No VRAM spill with 8 long prompts:** prefill is split per sequence in the hybrid sliding-window/recurrent
   memory, and new requests are admitted by the real number of free KV cells.
11. **Shared prompts are stored once:** with `-kvu`, a request whose prompt shares at least 512 tokens with another
   running request copies those KV cells and that request's recurrent state instead of processing them again.
   8 requests with a 5.7k shared prefix: 33.5 s to 6.9 s; 8 OpenCode subagents took a 13.1k system prompt in 7-42 ms
   each. `LLAMA_KV_FORK=0` turns it off, `LLAMA_KV_FORK_MIN` sets the threshold.
12. **No f16 copy of the quantized KV cache** (`fattn-banded.cu`, `fattn.cu`): the banded attention kernel dequantizes
   q4_0 rows while loading its tiles, and decode of up to 8 tokens uses the vector kernel that reads q4_0 directly.
   At 128k that copy was ~0.5 GB. The relative-position bias is also chunked so the compute buffer never grows past
   its startup size. VRAM at 8 requests: 11.81 to 11.38 GB, same speed.
13. **More fusions for parallel decode** (`inkling2.cu`, `topk-moe.cu`): the folded dynamic short convolution
   (window, state store, taps) is one kernel for any number of sequences, not just one; the MoE router fusion also
   covers the normalized top-k weights with their epsilon clamp. Checked in `test-backend-ops`
   (`INKLING2_CONV_DYN_FOLD`, `TOPK_MOE`).
14. **Speculative decoding tested, not used.** DSpark draft models load in this branch
   (`-md <draft.gguf> --spec-type draft-dspark`), but on this bandwidth-bound MoE every verified draft token pulls in
   more experts: 1 chat 72.6 tok/s with 2-token drafts vs ~80 without (58% accepted), 8 chats 58.8 vs ~195 total.

15. **One prompt batch per chat follow-up** (`server-context.cpp`): the context checkpoint for a hybrid
   SWA/recurrent model is taken after the prompt batch instead of splitting the batch in two before it, so a short
   follow-up costs one GPU pass. `LLAMA_SERVER_CKPT_END=0` restores the old split.
16. **CUDA graphs for prompt batches** (`ggml-cuda.cu`): graphs are keyed by batch shape and captured on first use,
   so a follow-up batch of a size seen before replays its graph (~15 ms less per follow-up). At most 4 graphs are kept
   (least recently used is dropped) so the cache costs no extra VRAM. `GGML_CUDA_GRAPH_SHAPE_KEY=0`,
   `GGML_CUDA_GRAPH_EAGER=0`, `GGML_CUDA_GRAPH_MAX=<n>` (0 = unlimited).
17. **Tokenizer cache** (`llama-vocab.cpp`): every request resends the whole chat, and tokenizing a 42k-token chat took
   ~160 ms. The text between special tokens is tokenized independently, so its tokens are cached per piece (64 MB,
   least recently used dropped first). The active chat stays cached; output is identical. `LLAMA_TOKENIZE_CACHE=0`
   turns it off.

18. **GPTQ-style quantizer** (`inkling-fast/quant/`): expert gate/up bits are placed per layer group by measured
   sensitivity, and each 256-column block's quantization error is compensated in the columns not yet quantized,
   using the layer's input covariance recorded by `llama-imatrix` (`IMAT_COV_DIR`). Same 9.24 GB, mean KLD
   0.0491 to 0.0428 (kld.py), same speed. Round 2 (`Inkling-2-Mini-fast-9.3-tuned.gguf`) adds learned per-layer
   block scales and a larger calibration set: 0.0391. Details and rejected variants in `quant/README.md`.
19. **Uncensored build** (`Inkling-2-Mini-Uncensored-fast.gguf`, the current file): AtomicChat's edit is a rank-1 LoRA
   on `attn_output`, `ffn_down_exps` and `token_embd`. It is merged into the Q8_K_XL weights at full precision (expert
   down: NVFP4 scale baked in, padded to 1280 rows), and only those tensors are re-quantized with the 9.3-tuned types;
   the GPTQ'd and tuned expert gate/up are reused unchanged (`quant/merge_lora.py`). KLD 0.0376 vs the merged Q8 reference (re-tuning the
   scales on top: 0.0386, not used).

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

Place `Inkling-2-Mini-Uncensored-fast.gguf` and the Q8 vision projector next to the scripts, or pass a folder
with -ModelDir.

~~~powershell
.\inkling-fast\run-inkling.ps1
.\inkling-fast\stop-inkling.ps1
~~~

The launcher starts one configuration: 128k shared context, 8 slots, q4_0 KV, vision, a 512-token microbatch and a
1,024-token image limit. realtest.py, vision loaded, desktop apps open (image rates include vision encoding and
prompt processing; decode is the first reply and a follow-up at about 9k tokens):

| Model | Photo / screenshot / paste (tok/s) | Decode first / follow-up (tok/s) |
| --- | ---: | ---: |
| Uncensored-fast | 855 / 1001 / 1842 | 82.5 / 73.5 |
| fast-9.3-tuned (previous, same run) | 843 / 973 / 1805 | 82.1 / 73.4 |

The Fast (64k) and Long (256k) modes were removed; `-Ctx 65536` still gives a smaller context. The server uses the
`inkling-2-mini` API alias. Other options include `-Parallel`, `-NoVision`, `-Foreground`, `-Port`, `-LocalOnly`,
`-CacheRam`, and `-Engine`.

Prefill uses implicit causal bounds for contiguous or position-ordered K/V and dequantizes q3_0/q4_0 tiles in shared memory.
Set `LLAMA_FA_KV_CHUNK=0` to compare against the dense-mask and staged-conversion paths.

Debug switches: `INKLING2_NO_FORK_FUSE=1` turns the glue fusions off, `INKLING2_FUSE_STATS=1` logs which ones fire.
