# Known issues

Status of version 0.1.0 on the reference machine. All of this was observed on one machine, so a rate on your machine can differ. Status values:

- **open**: no fix. A workaround may exist.
- **worked around**: a default setting of this release avoids the problem; the underlying defect remains.

Sources are files under [`docs/notes/`](notes/README.md), in Ukrainian. `known-defects` stands for `notes/measurements/known-defects.md`, and `mtp-notes` for
`notes/measurements/mtp-notes.md`. What to do when something fails is in [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Every user

| ID | Symptom | When | Status | What to do | Source |
|---|---|---|---|---|---|
| KI-1 | `hipErrorIllegalAddress` / `illegal memory access`, in the kernel log `[gfxhub] page fault … address 0x0 … UTCL2 client TCP` on the first card. The rank on that card dies and the other rank stays | At start, or on the first request. Seen on all models and both attention kernels. In the most systematic test: 5 of 89 starts failed at start, and 5 of 52 failed on the first request. On Qwen3.6-35B-A3B about one start in 3–4 failed this way in the series of 7 Oct 2026, so retries are normal there. Random, so one run proves nothing | open | `serve.sh` retries and checks the driver between attempts, and it sends a first request as a probe. Retries work only if the other rank exits. In the release check of 5 Nov 2026 it never did: 4 faults in 27 starts (Qwen3.5-35B-A3B on the first request; Qwen3.6-35B-A3B, Qwen3.8-27B and Qwen3.5-9B during startup), and each time the surviving rank kept running in `torch.cuda.synchronize` and ignored SIGTERM, also from `kill-stuck-rank.sh`, so `serve.sh` stopped with code 6 and no new attempt. A reboot cleared it every time. Two experiments reduced the failures: `AMD_SERIALIZE_KERNEL=3` gave 0 of 20 starts against 3 of 17 the same night, cost not measured, so the failure depends on timing; `VLLM_ROCM_TP_SYNC_VIA_ALLREDUCE=1` gave 0 first-request failures of 32 against 5 of 52, weak evidence, and one case of the fault happened without that code path. Root cause unknown | known-defects |
| KI-2 | The machine freezes; kernel log `workqueue: kfd_process_wq_release [amdgpu] hogged CPU for >10000us` | SIGKILL of a vLLM process with unfinished GPU page faults | worked around | Stop with `./stop.sh` and wait. If processes remain, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md#the-process-does-not-exit). Never use `kill -9` directly: `scripts/kill-stuck-rank.sh --force` first checks the kernel log for this rank and its card, and it reduces this risk without removing it. The fork does not send SIGKILL by default (`VLLM_ROCM_NO_SIGKILL=1`) | known-defects |
| KI-3 | `LLVM ERROR: Cannot select intrinsic llvm.amdgcn.fdot2.bf16.bf16` | `--dtype bfloat16`: Triton kernels need a bf16 dot instruction that gfx1030 does not have | worked around | Use `--dtype float16`, which `serve.sh` sets | known-defects, `vllm-gfx1030.md` |
| KI-4 | vLLM's own fast all-reduce assumes direct access to the other card | TP=2 without P2P: `torch.cuda.can_device_access_peer` returns false, while vLLM on ROCm does not check this | worked around | Pass `--disable-custom-all-reduce`, which `serve.sh` sets. The host all-reduce of this fork still runs | `vllm-gfx1030.md` |
| KI-5 | Start fails: `hipErrorOutOfMemory` already on `torch.zeros(1)` during PyNccl initialization | `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` | worked around | Do not set it | known-defects |
| KI-6 | Start refused: `To serve at least one request with the model's max seq len` | The window does not fit. A display on a compute card holds 0.35–0.69 GiB, and MTP takes more. At UTIL 0.86, MTP on 27B had a ceiling of 228,512 tokens | worked around | Lower `--ctx`, or raise `--util` if the card has free memory. The message states the limit. A display on the first card limits `--util` to about 0.978 | `vllm-gfx1030.md`, `perf-numbers` |
| KI-7 | Start fails: `Insufficient space in /dev/shm` | KV offload (`--offload-gb`). A stopped server leaves `/dev/shm/vllm_offload_*` | worked around | `serve.sh` removes files that no process holds. Before a manual start: `df -h /dev/shm` and `du -sh /dev/shm/vllm_*` | `offload-kv` |
| KI-8 | `TimeoutError: RPC call to sample_tokens timed out`, no page fault, ranks stay | `HIP_VISIBLE_DEVICES=1,0`: 3 of 3 starts hung on the first decode step. Cause unknown | worked around | Keep the ascending order. `serve.sh --gpus` refuses any other | known-defects |
| KI-9 | GPUs at 99 %, workers at about 300 % CPU, no progress. `/metrics`: `generation_tokens_total` stays at 1, `spec_decode_num_drafts_total` at 0 | The client aborts during prefill while MTP is on. Stuck at `num_accepted_tokens_event.synchronize()` | open | Set client timeouts longer than the longest prefill, or run with `--no-mtp` | known-defects, `mtp-notes` |
| KI-10 | Anyone who can reach the port can use the server | The server has no authentication | open | Keep the default `127.0.0.1`. `serve.sh --host 0.0.0.0` warns about this | |
| KI-11 | `EngineDeadError` after a 20 s wait in the host all-reduce. Reports a timeout and writes `hostar-rank*.txt` | One rank stalls inside the host all-reduce, for example after the other crashed. The hypothesis about SVM work in the driver during KV offload is not confirmed. A stuck rank has no clean exit on this machine | open | Stop normally. If ranks remain, reboot | known-defects |
| KI-12 | Wrong collective results on one rank in about half of process starts, fixed for the whole process. Logits exchanged this way give random text | RCCL `all_gather` and `reduce_scatter` on this pair | worked around | Keep `VLLM_ROCM_COLLECTIVES_VIA_ALLREDUCE=1`, the default. One clean run is not evidence | known-defects, `vllm-gfx1030.md` |
| KI-13 | EngineCore stays in a clean stop: futex wait, SIGTERM and SIGINT have no effect, workers have exited | Seen once in the MoE determinism series | open | Reboot when convenient. Do not send SIGKILL | known-defects |
| KI-14 | Long silence during prefill | Normal: the server sends nothing during prefill. Fixed logging shows a stats line with growing KV cache use | not a defect | Watch the log. Do not treat silence as a hang | known-defects |
| KI-38 | `rocminfo`: `HSA_STATUS_ERROR_OUT_OF_RESOURCES`; PyTorch: `No CUDA GPUs are available`; kernel log `amdgpu_device_ip_resume failed (-62)`; `power/runtime_status` of the card is `error` | Idle cards without a display enter runtime power saving, and tools that read sensors most likely wake them again. Seen once on 4 Nov 2026 on the card behind the chipset, during a long disk stall, with no vLLM running | open | A reboot restored the card. To keep idle cards awake until the next boot: `echo on \| sudo tee /sys/bus/pci/devices/<address>/power/control` | server check of this release |
| KI-39 | `GPU KV cache size` in the server log differs between starts of the same preset, so a load that fit yesterday can be preempted today | Seen on every preset that was started several times. Qwen3.5-35B-A3B: 584,782 to 712,493 tokens; with MTP 268,101 and 205,544, and 32k × 4 was preempted at the smaller value; Qwen3.8-27B: 580,663 and 617,306 | open | Read the line after each start and plan for the smaller value. The size follows from the memory left after startup (the line `Available KV cache memory`). Why that amount changes is not established | release check, 5 Nov 2026 |

## MTP

| ID | Symptom | When | Status | What to do | Source |
|---|---|---|---|---|---|
| KI-15 | Decode drops with MTP, for example from 216 to 153 tokens/s on 35B-A3B with 4 streams | Speculation silently rounds graph sizes up to multiples of 1 + K, so the default [1,2,4,8,16,24] becomes {3,6,9,18,24} | worked around | The presets pass explicit `cudagraph_capture_sizes`. In a manual command use multiples of 1 + K | known-defects, `mtp-notes` |
| KI-16 | The MTP gain falls with streams, and on some points MTP is slower | Likely cause, not measured separately: with 4 streams and 2 drafts a step has 12 tokens, above the 8 rows of `skinny_int4_ext`, so each W4A16 linear layer runs twice per step. See [the matrices](PERFORMANCE.md#full-matrices) | open | One stream: `--mtp`. For several streams measure your load: on Qwen3.6-35B-A3B with 4 agent clients the gain was 5.6 % | `2026-09-29-night-results.md`, `2026-10-07-qwen36-config-results.md` |
| KI-17 | HTTP 400: `The min_p and logit_bias sampling parameters are not yet supported with speculative decoding` | `min_p > 0` or `logit_bias` in a request while MTP is on | open | Do not send them while MTP is on | `mtp-notes` |
| KI-18 | `ValueError: There is no module or parameter named 'fc.weight' in Qwen3_5MultiTokenPredictor` | Your own checkpoint where the MTP head was quantized although its weights are BF16 | worked around | Add `mtp.fc`, `mtp.layers.0.mlp.{down,gate,up}_proj` and `mtp.layers.0.self_attn.{k,o,q,v}_proj` to `quantization_config.ignore`. `scripts/add-mtp-head.py` does this for Qwen3.5-9B | known-defects |
| KI-19 | Qwen3.5-9B: `--mtp` refused | The AWQ repository has no MTP head | worked around | `scripts/add-mtp-head.py <model dir>` fetches the 15 `mtp.*` tensors from `Qwen/Qwen3.5-9B` | known-defects |
| KI-20 | llama.cpp comparison has no MTP row | llama.cpp with MTP did not start at window 262,144: out of memory on the draft KV cache | not a defect of this fork | See [PERFORMANCE.md](PERFORMANCE.md#against-llamacpp) | `2026-10-07-qwen36-config-results.md` |

## Text quality and sampling

| ID | Symptom | When | Status | What to do | Source |
|---|---|---|---|---|---|
| KI-21 | Fragments and word collisions in non-English text, measured on Ukrainian, at `t=1.0, top_k=20, top_p=0.95` on the 27B preset. Greedy decoding is clean | Quantization: on identical weights, the engines (this fork and llama.cpp) showed no significant difference, while replacing AWQ group 128 with RTN group 32 improved the metric ×3.7 (p = 0.0005) | open | Use weights quantized with group 32 if you need it. The 27B and 9B presets set `VLLM_ROCM_MIN_P_FLOOR=0.05`, which acts without MTP | `vllm-gfx1030.md` |
| KI-22 | Loops inside long reasoning | **A temperature ceiling.** `VLLM_ROCM_MAX_TEMPERATURE=0.3` improved a metric on short answers but caused loops on long reasoning, so it was cancelled | worked around | Keep `VLLM_ROCM_MAX_TEMPERATURE=0`, as in all presets | `vllm-gfx1030.md` |
| KI-23 | One turn spends its whole thinking budget and ends without an answer or tool call | `thinking_token_budget` without a reasoning end string | worked around | Keep the preset's `REASON_END` | `vllm-gfx1030.md` |
| KI-24 | Out of memory on a long prefill chunk: 1.89 GiB requested with 768 MiB free | Prompt logprobs (`echo`) with TP=2 | open | Lower the prefill chunk: `-- --long-prefill-token-threshold 512` | known-defects |

## KV offload to host memory

| ID | Symptom | When | Status | What to do | Source |
|---|---|---|---|---|---|
| KI-25 | Requests wait in `Deferred` for up to an hour; with 4 streams only 3 progress | `--offload-gb` above 0 with several parallel sessions. The transfer queue is serial | open | Limit concurrency or disable offload for parallel sessions | `offload-kv`, known-defects |
| KI-26 | `AssertionError` in `kv_transfer/kv_connector/v1/offloading/config.py` (`tokens_per_chunk % tokens_per_block`) | A fixed `block_size` in the connector settings | worked around | Do not set `block_size`. `serve.sh` does not | `offload-kv` |

## Pipeline parallelism

`serve.sh` runs only tensor parallelism. Pipeline parallelism needs a manual `vllm serve` command, and it is covered less: see [the comparison](PERFORMANCE.md#against-pipeline-parallelism-on-three-cards).

| ID | Symptom | When | Status | What to do | Source |
|---|---|---|---|---|---|
| KI-27 | Start refused: `Speculative decoding with pipeline parallelism is blocked on this fork`. With the block lifted, output was corrupted | Pipeline parallelism and MTP together | open | Use one or the other. `VLLM_ALLOW_PP_SPEC=1` is for debugging only | known-defects |
| KI-28 | Hang on the first request: `No available shared memory broadcast block found in 60 seconds` | Pipeline parallelism with async scheduling | worked around | Add `--no-async-scheduling` | known-defects |
| KI-29 | Start fails: `NCCL error: unhandled cuda error`; kernel log `PCIE atomic ops is not supported` | A card on a chipset port without PCIe AtomicOps | worked around | `VLLM_PP_HOST_TRANSFER=1`, or move the card to a CPU root port | known-defects |
| KI-30 | `torch.OutOfMemoryError` in `w4a8_apply`, 474 MiB | 27B on three cards, UTIL 0.97, four simultaneous prefills | worked around | UTIL 0.93 for four streams | known-defects |
| KI-31 | `torch.OutOfMemoryError: Tried to allocate 352.00 MiB` in `moe_gemm` | Gemma 4 on three cards, four long prompts. The working buffer depends on routing and is not reserved by profiling. TP=2 passes | open | Reduce the concurrent long prompts. Lower UTIL is not verified | known-defects |
| KI-32 | Page fault with `UTCL2 client ID: SDMA0`, then `EngineDeadError` | Seen once: profiling on three cards with 4 × 125,000 tokens | open | None verified | known-defects |
| KI-33 | Decode with 3–4 streams reaches only ×2.1–2.2 of one stream | Three cards, Qwen3.8-27B | open | None | known-defects |
| KI-34 | Wrong GEMM split for K above 16384: chunk 1 on `down_proj` of 27B | Pipeline with TP=1, K = 17408. TP=2 is not affected. Whether this affects results is not confirmed | open | None | known-defects |

## Diagnostics and experimental switches

| ID | Symptom | When | Status | What to do | Source |
|---|---|---|---|---|---|
| KI-35 | `api registration failed with error code 16` | `rocprofv3` with PyTorch from a venv | open | Use `torch.profiler`. Under full CUDA graphs its times are too large, so confirm with an A/B of a switch | known-defects |
| KI-36 | Wrong output without any error | `VLLM_ROCM_REPLICATE_VOCAB=2` with tied embeddings; `VLLM_ROCM_LMHEAD_INT8=1` with a `float32` head; `VLLM_ROCM_SKIP_ALL_REDUCE=1`, `VLLM_ROCM_ABLATE` | open | Do not use these for serving. All are off by default. See [SWITCHES.md](SWITCHES.md) | known-defects |
| KI-37 | vLLM's skinny GEMMs (`wvSplitK`, `LLMM1`) return wrong results on gfx1030, relative error 0.46–0.62 | gfx10 | worked around | They stay disabled for gfx10. The cause was not found | `vllm-gfx1030.md` |

## Fixed in this release

Compared with upstream vLLM or with earlier builds of this fork.

| Problem | Fix | Source |
|---|---|---|
| MTP broke text under random sampling at TP=2: 0 of 16 good runs, 16 of 16 without MTP | The chosen tokens are exchanged between the ranks before the recurrent state advances. 16 of 16 after the fix | `mtp-notes`, [ARCHITECTURE.md](ARCHITECTURE.md#mtp-on-hybrid-models) |
| SIGKILL of the workers at every stop, because `--shutdown-timeout` is 0 by default | `VLLM_ROCM_NO_SIGKILL=1`, and `serve.sh` passes `--shutdown-timeout 60` | known-defects |
| No log lines during a long prefill | The stats line appears during prefill | known-defects |
| Gemma 4 died on the first request: `Both operands must be same dtype. Got int8 and fp32` (sliding-window layers, int4 KV) | Fixed in the int4 attention path | known-defects |
| Gemma 4 with the HIP decode kernel died on the first request: the segment buffer was sized for head 512 and read with a layout for head 256 | Commit `7a4b1f450e`. Then measured on this path | known-defects, `2026-09-29-gemma4-26b-a4b.md` |
| The HIP decode kernel wrote NaN at one query head per KV head | Compiler lowering of an LDS read, see the comment in `int4_decode_attn.hip` | known-defects |
| The same request gave different tokens on the MoE kernel (KL 0.041) | `VLLM_ROCM_MOE_RDNA2_DETERMINISTIC=1`: KL 0 on a repeat. No speed cost was measured on Gemma 4 | known-defects |
| `RuntimeError: rotary_embedding …` during MTP profiling on a Qwen3.5 text model | The draft config no longer keeps mRoPE sections | known-defects |
