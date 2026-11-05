# Architecture: what this fork changes in vLLM and why

The fork is one patch against vLLM `c296851a7d`: 90 files, 29 of them new (`patches/MODIFIED_FILES.txt`). Everything is behind checks for gfx10 or
behind environment switches, listed in [SWITCHES.md](SWITCHES.md). The code comments in the patch are in Ukrainian and refer to the working notes.
Measurements below were taken on one RX 6800 or on the RX 6800 + RX 6800 XT pair at 1600 MHz; dates and sources are in
[docs/notes/](notes/README.md).

## Map

| Area | Main files | Switch |
|---|---|---|
| Platform detection | `vllm/platforms/rocm.py` | always on for gfx10 |
| All-reduce through host memory | `csrc/rocm/host_all_reduce.cu`, `vllm/distributed/device_communicators/host_all_reduce.py` | `VLLM_ROCM_HOST_ALL_REDUCE` |
| Collectives built from all-reduce | `vllm/distributed/device_communicators/base_device_communicator.py` | `VLLM_ROCM_COLLECTIVES_VIA_ALLREDUCE` |
| Token agreement between TP ranks | `vllm/v1/worker/gpu_model_runner.py` | `VLLM_ROCM_TP_SYNC_SAMPLED` |
| W4A16 linear layers | `vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py`, `skinny_int4_ext.hip` | `VLLM_ROCM_SKINNY_EXT` |
| W4A8 and W4A4 linear layers | `vllm/model_executor/kernels/linear/mixed_precision/w4a4/w4a4_ext.hip` | `VLLM_ROCM_W4A8`, `VLLM_ROCM_W4A4` |
| W8A16 layers | `vllm/model_executor/layers/w8a16_skinny.hip`, `w8a16_skinny.py` | `VLLM_ROCM_W8A16` |
| Matrix-vector kernel for `lm_head` | `vllm/model_executor/layers/rdna2_matvec.py` | `VLLM_ROCM_SMALLN_FIX`, `VLLM_ROCM_LMHEAD_INT8` |
| int4 KV cache, attention | `vllm/v1/attention/ops/int4_per_token_head.py`, `int4_decode_attn.hip`, `int4_prefill_attn.hip` | `VLLM_ROCM_INT4_*` |
| MoE experts | `vllm/model_executor/kernels/moe/moe_q_gemm_rdna2.hip` | `VLLM_ROCM_MOE_RDNA2` |
| Gated DeltaNet decode | `vllm/model_executor/kernels/gdn_fused/gdn_fused_ext.hip` | `VLLM_ROCM_GDN_FAST_DECODE` |
| Pipeline parallelism through host | `vllm/distributed/parallel_state.py`, `vllm/v1/core/sched/scheduler.py` | `VLLM_PP_*` |
| Shutdown without SIGKILL | `vllm/v1/utils.py`, `vllm/v1/executor/multiproc_executor.py` | `VLLM_ROCM_NO_SIGKILL` |

## One root cause: gfx10 belongs to no GPU family

vLLM selects kernels and their tuning by GPU family, in `vllm/platforms/rocm.py`. The checks match `gfx11` and `gfx12`, then `gfx9`, then CDNA.
gfx10 matches none, so every branch that depends on these flags falls through to the most generic path. Nothing fails with an error message. On RDNA2
this meant:

- the W4A16 GEMM used by every linear layer got tile sizes tuned for MI300, which has 304 CUs and 64-wide waves;
- the fast HIP decode path for quantized weights was disabled;
- unquantized GEMMs such as `lm_head` went to hipBLASLt with tiles meant for large matrices;
- the vision encoder took the SDPA math path and tried to allocate 128 GiB.

Fixing the detection was the largest single step in prompt processing: most of the initial gap to llama.cpp in that phase came from this check and not from the hardware. In decode the largest step was captured graphs instead of eager mode (5.06 to 13.70 tokens/s, see `PERFORMANCE.md`).

## Where the time goes

For a kernel with *W* operations and *Q* bytes moved, the roofline model gives the attainable performance `min(π, β · W/Q)`. *π* is the compute peak and *β*
the memory bandwidth. A linear layer with *K* inputs, *N* outputs and *b*-bit weights processing *M* tokens has an intensity of about `16M/b` operations per byte.

- **Prefill.** *M* is thousands of tokens, so the layer is limited by compute.
- **Decode.** *M* is the number of running sequences. With 4-bit weights the intensity is about `4M` operations per byte, against a ridge point of about 45 for fp16 on an RX
  6800 (23.2 TFLOPS / 512 GB/s).

In a decode step of Qwen3.5-9B (4-bit GGUF weights, one RX 6800, TP=1, eager mode, 31 Aug 2026), the linear layers read 5.38 GB of weights in 12.51 ms: 430.6 GB/s, or 84 % of the peak. They are close to the limit. The attention
kernel read its cache at only 30 GB/s, 5.9 % of the peak, which is why it got its own kernel.

## Tensor parallelism without P2P

### All-reduce through host memory

The pair has no P2P access: `hipDeviceCanAccessPeer` is 0 in both directions. RCCL therefore uses its SHM transport, GPU to host memory to GPU, with CPU proxy
threads, and the smallest exchange costs about 150 µs. Tensor parallelism on Qwen3.8-27B makes about 130 exchanges per token.

`csrc/rocm/host_all_reduce.cu` does the same inside one kernel, without the CPU:

1. Rank 0 creates a 2 MiB shared-memory segment. It is registered with the GPU through `hipHostRegister`, so the card accesses it uncached.
2. Each rank writes its vector into its own slot.
3. It publishes its step number.
4. It waits for the other rank's step number.
5. It adds the other rank's vector to its own.

Details that matter:

| Detail | Why |
|---|---|
| The step number lives in memory and not in a kernel argument | Arguments are frozen when a CUDA graph is captured, so an external counter would stay the same on every replay |
| Data is double-buffered by the parity of the step | A rank that runs one step ahead must not overwrite data that the other has not read |
| A magic word at the end of the segment | Rank 1 must not join a segment from a crashed run, or start before rank 0 has cleared the memory |
| The segment name is agreed through a CPU group (gloo) | RCCL `broadcast` hung on this pair. A random suffix separates runs |
| Up to 256 Ki fp16 elements per exchange | Above this RCCL is as fast or faster, so larger exchanges go through it |
| Exactly two ranks, fp16, contiguous tensors | Everything else goes through RCCL |
| The decision is agreed between ranks at start | If one rank took this path and the other RCCL, the first would wait forever |
| A 20 s spin limit, then an abort | Instead of a silent hang; the rank writes its recent calls to `VLLM_ROCM_HOST_AR_DUMP_DIR` |
| An in-memory ring of recent calls, written to disk only on failure | Earlier tracing shifted the timing and made the failure disappear |

Results are in [PERFORMANCE.md](PERFORMANCE.md#all-reduce). **Note:** the header comment of the kernel says that exchanges are half of the decode time. A later measurement
disproved this: all exchanges together are 8 % of a step. The comment was kept to keep the release tree equal to the measured code. vLLM's own fast all-reduce must stay off
(`--disable-custom-all-reduce`); `serve.sh` passes this flag. The host all-reduce works with this flag on.

### Other collectives

On this pair, RCCL `all_gather` and `reduce_scatter` return wrong data on one rank in about half of process starts. The outcome is decided when the communicator is created
and stays for the whole process, so one successful run proves nothing. Logits exchanged through them produced random text. `send`/`recv` are reliable. With
`VLLM_ROCM_COLLECTIVES_VIA_ALLREDUCE=1` the collectives are built from all-reduce instead.

### Token agreement between ranks

Each TP rank samples with its own random generator. `VLLM_ROCM_TP_SYNC_SAMPLED=1` makes every rank adopt the token chosen by rank 0. Without it, 1,111 of 4,000 steps diverged
in a test, against 0 with it.

## Kernels for linear layers

### Tile sizes

Every linear layer goes through the Triton W4A16 GEMM. Measured on one RX 6800, *K* = 5120, group size 128, *M* = 2048:

| Tile M × N × K | N = 5120 | N = 17408 |
|---|---:|---:|
| 128 × 128 × 32 (the MI300 branch, used before) | 1.88 TFLOPS | 1.84 TFLOPS |
| 128 × 32 × 64 (the RDNA 3.5 branch) | 3.77 TFLOPS | 3.56 TFLOPS |
| **32 × 128 × 32 (chosen)** | **9.43 TFLOPS** | **9.53 TFLOPS** |

A 128 × 128 tile does not fit into the 256 vector registers of a thread and spills to memory. This change alone took prompt processing on Qwen3.8-27B from 70 to 267.8 tokens/s.

### Dot-product instructions

RDNA2 has no matrix units; they first appeared in RDNA3. Its fastest arithmetic is `v_dot2_f32_f16`, `v_dot4_i32_i8` and `v_dot8_i32_i4`, which add 2, 4 or 8 products to a
32-bit accumulator. The peak is `f · N_CU · 64 · 2n`, and the measured ceilings on one RX 6800 (`kernels/dot-peak.hip`) follow it: 23.2 TFLOPS for fp16, 46.5 TOPS for int8,
92.6 TOPS for int4. hipBLASLt fp16 on the same shape gives 20.2 TFLOPS.

### int4 GEMM

A W4A4 GEMM written in HIP (`kernels/int4-gemm.hip`) went through these accepted steps, in TOPS on *M* = 2048, *K* = 5120, *N* = 17408: 9.6 first version; 29.1 inner loop; 31.4 double
buffer; 35.7 wide loads; 36.6 one barrier; 37.1 two groups per load; 50.0 group scales in LDS; 52.1 paired LDS reads. That is 56 % of the instruction ceiling and ×2.58 hipBLASLt fp16.
Unrolling the inner loop, `ds_read_b128` and an 8 × 8 micro-tile were rejected by measurement. Every variant checks its result against direct computation on 12 random elements: twice a
"fast" kernel was fast because it computed the wrong thing.

### Activation precision

| Format | Kernel | TOPS | Status |
|---|---|---:|---|
| W4A16 | Triton, tile 32 × 128 × 32 | 9.5 | Default for decode |
| W4A8 | HIP, signed int8 activations, `v_dot4` | 25.3 | Used in prefill by the 27B and 9B presets (`VLLM_ROCM_W4A8=256`) |
| W4A4 | HIP, int4 activations, `v_dot8`, scale per group | 49.3 | Off: on 9B, perplexity rose by 19.9 %, the cheapest variant still 6.2 % |
| fp16 | hipBLASLt, unquantized, for reference | 20.2 | |

### Matrix-vector kernel

In decode, a linear layer multiplies a matrix by one or a few vectors, so its time is the time to read the weights. For `lm_head` of the 27B model in fp16, with a memory roofline of 3.02 ms
per step:

| Kernel | Time, ms | Share of the roofline | Relative error |
|---|---:|---:|---:|
| hipBLASLt (tile for large M) | 16.44 | 18 % | |
| `wvSplitK` (vLLM) | 2.75 | | **0.62: wrong** |
| `LLMM1` (vLLM) | 2.59 | | **0.46: wrong** |
| **`rdna2_matvec`** | **3.38** | **89 %** | 3.3 · 10⁻⁴ |

vLLM's own skinny kernels compile and run faster than the roofline allows on gfx1030, because they compute a wrong result. The cause was not found, so they stay disabled for gfx10. A kernel
that beats the roofline is a bug.

### Unquantized layers and MoE experts

- `VLLM_ROCM_W8A16=1` quantizes unquantized linear layers to int8 with a per-row scale at load time, and decodes through the HIP kernel `w8a16_skinny`: TG ×1.18–1.23 on Qwen3.6-35B-A3B.
- `VLLM_ROCM_MOE_RDNA2=1` uses a native HIP kernel for compressed-tensors W4A16 MoE experts, ported from the RDNA3 path of upstream. By default it sums partial results in fp32 in a fixed order,
  because atomic fp16 accumulation gave different tokens between runs (KL 0.041, top-1 agreement 93.1 %). The repeat then gives KL 0.
- `VLLM_ROCM_GEMMA4_FUSE_FFN_AR=1` joins the two reductions of the dense and MoE parts of Gemma 4 into one, 30 fewer all-reduces per step: +0.8–2 %.

## Attention and the 4-bit KV cache

A full-attention layer computes the scores `S = Q·Kᵀ/√d` and the output `O = softmax(S)·V`. In the new kernels both products are integer: `Q·Kᵀ` through the W4A8 kernel, with keys in 4 bits and
queries quantized to int8, and `P·V` with the probabilities quantized to int8 per row. The KV cache keeps one scale per token and head, after a randomized Hadamard transform of keys and values
(`VLLM_ROCM_INT4_RHT`).

| KV cache and attention products | Prefill of 74,000 tokens, tokens/s |
|---|---:|
| int8 KV, products in fp16 (before) | 162.4 |
| int8 KV, integer products | 270.5 |
| **int4 KV, integer products** | **273.4** |

Qwen3.8-27B with UD-Q4_K_XL GGUF weights, W4A8, TP=2, window 163,840, 31 Aug 2026. The server had loaded these weights and not AWQ, as the section of 1 Sep in `docs/notes/vllm-gfx1030.md` records, so the comparison of formats holds and the absolute numbers are those of the GGUF. At equal speed the int4 cache holds almost twice as many tokens: 379,562 against 203,337 on 31 Aug, 324,949 against 175,542 on 1 Sep. The memory estimate of vLLM varies between runs, so these capacities are not exact. Quality cost on 9B: perplexity 5.965 with int4 KV against 5.940 with int8, +0.4 %.

Later HIP kernels:

- **Decode attention** (`int4_decode_attn.hip`): cuts the context-dependent part of a step by ×3.3, from 86.5 to 26.3 µs per 1,000 context tokens (`docs/notes/measurements/2026-09-28-mtp-attention-results.md`). TG ×1.17–1.36 on Qwen3.8-27B.
- **Prefill attention** (`int4_prefill_attn.hip`): PP ×1.62–2.18 on Gemma 4, ×1.32 on Qwen3.6.

Prefill still slows down with the prompt length, because the cost of attention grows quadratically with the context. This is the main open item for long prompts.

## MTP on hybrid models

Qwen3.8-27B has an MTP head that drafts 2 tokens, and the full model verifies them in one step. With sampling at temperature 1.0, generation in Ukrainian broke: 0 of 16 runs were good with MTP,
16 of 16 without it.

A draft token is accepted at random, and each TP rank had its own generator, so the ranks accepted different numbers of tokens on 23.0 % of steps. The 48 linear-attention layers advanced their
recurrent state by that number before the ranks exchanged the chosen tokens. A KV cache survives such a mismatch, because rejected positions are overwritten. A recurrent state does not, because
it accumulates. The fix sends the chosen tokens between the ranks before the state is advanced: 16 of 16 good runs. Greedy decoding hid the bug, because it does not use the random generator.

Other MTP details:

- Speculation changes the sizes of the captured graphs without warning: the sizes are rounded up to multiples of 1 + K. The presets therefore pass explicit `cudagraph_capture_sizes`.
- `VLLM_ROCM_INT4_DECODE_QLEN_MAX=3` lets the HIP decode kernel handle the three query tokens of a verification step.
- MTP with pipeline parallelism corrupts the output, so the start is refused unless `VLLM_ALLOW_PP_SPEC=1` is set for debugging.

## Pipeline parallelism on three cards

The third card of the reference machine sits on a chipset PCIe 2.0 x1 slot without PCIe AtomicOps, where RCCL kernels do not start. `VLLM_PP_HOST_TRANSFER=1` moves the stage tensors through host
memory and a gloo group. The scheduler cap `VLLM_PP_MICROBATCH_SPLIT` lets the stages overlap. Pipeline parallelism gave more KV cache and less decode speed than TP=2
([comparison](PERFORMANCE.md#against-pipeline-parallelism-on-three-cards)), so TP=2 is the recommended mode.

## Shutdown

Upstream vLLM sends SIGKILL to its workers at every stop, because the default of `--shutdown-timeout` is 0. A SIGKILL on a process with pending GPU address faults can hang `kfd_process_wq_release` in the
driver and freeze the machine. `VLLM_ROCM_NO_SIGKILL=1` (default) skips forced kills and logs the surviving processes. `serve.sh` passes `--shutdown-timeout 60`, and `stop.sh` and the systemd unit from
`scripts/install-service.sh` use only SIGTERM.

## What did not work

- **Profiler attribution.** Collectives looked like 49 % of decode time under RCCL; removing them, with the host all-reduce in place, showed 8 %. Two optimizations were built on the wrong diagnosis.
- **vLLM's skinny GEMMs** compute wrong results on gfx1030. The cause is unknown.
- **W4A4** is fast but costs too much accuracy.
- **MTP with several streams.** A kernel for up to 8 rows (`skinny_int4_ext`) removed the loss with 2 streams, where a step has 6 tokens. With 4 streams a step has 12 tokens, so each W4A16 linear layer runs twice per step, which is the likely cause of the smaller gain there. On an agent workload with 4 clients the gain is 5.6 %.
- **llama.cpp comparisons on different days.** Earlier two-card numbers used other prompts and software versions. Only the same-day comparisons are valid.

More: [docs/notes/](notes/README.md), which keeps every problem, its root cause, the measurements and what was rejected.
