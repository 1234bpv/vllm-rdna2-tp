# Environment switches

The fork adds 83 environment variables to `vllm/envs.py`. Most users need none of them: the presets set the ones that matter ([CONFIGURATION.md](CONFIGURATION.md)).
Leave everything else at the default, and change one switch at a time when measuring.

- `Default` is the value used when the variable is unset. `unset` means that the code gets `None`.
- `Evidence` gives the entry in `vllm/envs.py` and where the value is read, as line numbers in the patched tree of version 0.1.0.
- `unclear` means that the effect could not be established from this code.

## 1. Host all-reduce and TP collectives

| Variable | Default | What it does | Set it when | Evidence |
|---|---|---|---|---|
| `VLLM_ROCM_HOST_ALL_REDUCE` | `1` | Runs small tensor-parallel all-reduces through pinned host memory instead of RCCL. | leave default; `0` only for an A/B run against RCCL. | `vllm/envs.py:1488`; `vllm/distributed/device_communicators/host_all_reduce.py:35` |
| `VLLM_ROCM_HOST_AR_THREADS` | `1024` | Sets the block width of the host all-reduce kernel. The C++ side reads the raw variable. An unset value, or a value below 64 or above 1024, falls back to 1024, rounded down to a multiple of 64. | leave default. | `vllm/envs.py:1600`; `csrc/rocm/host_all_reduce.cu:301-307` |
| `VLLM_ROCM_HOST_AR_DUMP_DIR` | `"/tmp"` | Directory where host all-reduce writes its ring buffer of recent calls when it aborts. | leave default. | `vllm/envs.py:1604`; `vllm/distributed/device_communicators/host_all_reduce.py:131` |
| `VLLM_ROCM_COLLECTIVES_VIA_ALLREDUCE` | `1` | On ROCm (the envs comment says gfx103x only), builds all_gather, reduce_scatter and broadcast from all_reduce, because RCCL 2.27.7 returns wrong data for them on one rank and hangs on broadcast. | leave default; `0` only to test RCCL collectives on another RCCL build. | `vllm/envs.py:1477`; `vllm/distributed/device_communicators/base_device_communicator.py:46-50` |
| `VLLM_ROCM_FP32_ALLREDUCE` | `0` | Casts fp16 and bf16 tensors to fp32 for the all-reduce sum, then casts the result back to the model dtype. | leave default; `1` only as a precision test, because it changes rounding. | `vllm/envs.py:1642`; `vllm/distributed/device_communicators/cuda_communicator.py:298-302` |
| `VLLM_ROCM_REPLICATE_VOCAB` | `0` | Replicates the embedding and/or lm_head on every TP rank instead of sharding them: `1` both, `2` lm_head only, `3` embedding only. Only the Qwen3.5 model file reads it. | leave default; `2` or `3` only to separate the two effects of vocab sharding. The envs comment gives about 1.2 GiB extra per card per replicated layer. | `vllm/envs.py:1497`; `vllm/model_executor/models/qwen3_5.py:261`, `:361` |
| `VLLM_ROCM_TP_SYNC_SAMPLED` | `1` | After sampling, every TP rank adopts the token chosen by rank 0, so ranks cannot continue with different texts. | leave default; `0` only to reproduce the divergence. | `vllm/envs.py:1658`; `vllm/v1/worker/gpu_model_runner.py:3907` |
| `VLLM_ROCM_TP_SYNC_VIA_ALLREDUCE` | `0` | Chooses the transport for the token broadcast above. `0` uses the process-group RCCL broadcast. `1` builds it from all_reduce through pynccl, on gfx10 with `VLLM_ROCM_COLLECTIVES_VIA_ALLREDUCE=1`. | leave default; `1` only as a workaround tried for the first-request fault. Per the docstring of `_tp_sync_sampled_tokens`, whether native broadcast is reliable on this setup is not established. | `vllm/envs.py:1667`; `vllm/v1/worker/gpu_model_runner.py:3922`, `:3925` |
| `VLLM_ROCM_GEMMA4_FUSE_FFN_AR` | `1` | For Gemma 4 with TP>1 and an MoE or second MLP block, sums the dense-MLP and MoE partial outputs in one all-reduce instead of two. The comment cites 30 fewer all-reduces per decode step. | leave default; `0` restores the upstream two-reduce order. The comment says the fused order is bit-identical to the separate reduces only at TP=2. | `vllm/envs.py:1731`; `vllm/model_executor/models/gemma4.py:616` |

## 2. Int4 prefill and decode kernels (and shared attention kernel knobs)

Rows marked "shared" apply to the Triton unified attention kernel, which serves both int4 and int8 KV.

| Variable | Default | What it does | Set it when | Evidence |
|---|---|---|---|---|
| `VLLM_ROCM_INT4_PREFILL_KERNEL` | `"hip"` | Chooses the kernel for 2D prefill attention over int4 KV. `"hip"` runs the fork's HIP kernel; any other value runs Triton. | leave default; `triton` only for A/B. | `vllm/envs.py:1710`; `vllm/v1/attention/ops/int4_per_token_head.py:1055` |
| `VLLM_ROCM_INT4_PREFILL_WARPS` | `4` | Warps per workgroup for the int4 prefill attention launch. | leave default. | `vllm/envs.py:1715`; `vllm/v1/attention/ops/int4_per_token_head.py:1282` |
| `VLLM_ROCM_INT4_PREFILL_WAVES_PER_EU` | `0` | Compiler hint `waves_per_eu` for the int4 prefill launch. `0` passes no hint. | leave default; tune only in a kernel benchmark. | `vllm/envs.py:1718`; `vllm/v1/attention/ops/int4_per_token_head.py:1283-1284` |
| `VLLM_ROCM_INT4_DECODE_KERNEL` | `"new"` | Chooses the 3D decode attention kernel over int4 KV. `"new"` runs the fork's HIP kernel; any other value runs Triton. | leave default; `triton` only as the reference in A/B runs. | `vllm/envs.py:1743`; `vllm/v1/attention/ops/int4_per_token_head.py:1008` |
| `VLLM_ROCM_INT4_DECODE_QLEN_MAX` | `1` | Largest query length per sequence that the HIP decode kernel accepts (1 to 4). `1` means plain decode only. `0` turns the HIP decode kernel off. Values above 4 send those steps to Triton. | leave default; raise to `3` only for measured MTP speculative decoding with two draft tokens. | `vllm/envs.py:1750`; `vllm/v1/attention/ops/int4_per_token_head.py:1007` |
| `VLLM_ROCM_INT4_DECODE_WARPS` | `2` | Warps per workgroup for the int4 3D decode launch. | leave default. | `vllm/envs.py:1723`; `vllm/v1/attention/ops/int4_per_token_head.py:1286` |
| `VLLM_ROCM_INT4_DECODE_WAVES_PER_EU` | `1` | Compiler hint `waves_per_eu` for the int4 decode launch. `0` passes no hint. | leave default. | `vllm/envs.py:1726`; `vllm/v1/attention/ops/int4_per_token_head.py:1287-1288` |
| `VLLM_ROCM_INT4_STAGES` | `0` | Forces the software-pipeline stage count of the int4 attention launch. `0` keeps the built-in rule, which the envs comment states is one stage. | leave default. | `vllm/envs.py:1706`; `vllm/v1/attention/ops/int4_per_token_head.py:1279` |
| `VLLM_ROCM_INT4_FP16_DOT` | `1` | In the int4 kernel inner loop, uses fp16 dot-product instructions when the query is fp16. `0` uses fp32 multiply-add. | leave default. | `vllm/envs.py:1755`; `vllm/v1/attention/ops/int4_per_token_head.py:1154` |
| `VLLM_ROCM_INT4_W4A8` | `1` | Enables the int4 attention path with int8-quantized queries and an integer Q·K product. Takes effect only when the query is fp16. | leave default; `0` only for A/B. | `vllm/envs.py:1853`; `vllm/v1/attention/ops/int4_per_token_head.py:1340` |
| `VLLM_ROCM_INT4_QBITS` | `8` | Bit width of the quantized queries in the int4 W4A8 path. `8` is the working mode; `6` and `4` are quality probes that do not change speed. | leave default. | `vllm/envs.py:1859`; `vllm/v1/attention/ops/int4_per_token_head.py:1350` |
| `VLLM_ROCM_INT4_RHT` | `1` | Applies a randomized Hadamard transform to queries, and to keys and values before the int4 cache write, and inverts it on the output. | leave default; `0` only as a measurement, since the envs comment says quality collapses without it. | `vllm/envs.py:1862`; `vllm/v1/attention/ops/int4_per_token_head.py:1220`, `:1329`, `:1401` |
| `VLLM_ROCM_INT4_RHT_FUSED` | `1` | Uses a fused transform matrix (signs and normalisation inside, computed in the input dtype). The fused path is taken only when `x.is_cuda` is true, head size is at most 1024, and dtype is fp16, bf16 or fp32. | leave default. | `vllm/envs.py:1865`; `vllm/v1/attention/ops/int4_per_token_head.py:1671` |
| `VLLM_ROCM_INT4_PV_INT8` | `1` | Quantizes the softmax probabilities per row to int8 and computes P·V in integers in the int4 kernel. `1` covers prefill only; `2` also covers decode. Takes effect only when the int4 W4A8 path is active. | leave default; `0` for quality or reproduction runs. | `vllm/envs.py:1541`; `vllm/v1/attention/ops/int4_per_token_head.py:1164-1165` |
| `VLLM_ROCM_INT4_TILE` | `0` | Overrides the int4 prefill tile size (must be a power of two). `0` keeps the default tile of 32. | leave default. | `vllm/envs.py:1585`; `vllm/v1/attention/ops/triton_unified_attention.py:956-972` |
| `VLLM_ROCM_INT8_PV_INT8` | `1` | Shared: integer P·V product in the Triton attention kernel for int8-per-token-head KV. `1` prefill only; `2` also decode. | leave default; `0` for A/B. | `vllm/envs.py:1569`; `vllm/v1/attention/ops/triton_unified_attention.py:1431`, `:1531-1532` |
| `VLLM_ROCM_QK_INT8` | `1` | Shared: integer Q·K product (int8 queries and keys) for int8 KV. Requires `VLLM_ROCM_INT8_PV_INT8 > 0`. `1` prefill only; `2` also decode. | leave default; `0` for A/B. | `vllm/envs.py:1581`; `vllm/v1/attention/ops/triton_unified_attention.py:1430-1432` |
| `VLLM_ROCM_ATTN_TILE` | `0` | Shared: overrides the tile size of the Triton attention kernel on gfx10. Setting it bypasses the automatic rule, which sizes the tile to the 64 KiB LDS limit. | leave default; set only in a measured tile sweep. | `vllm/envs.py:1536`; `vllm/v1/attention/ops/triton_unified_attention.py:1381-1396` |
| `VLLM_ROCM_ATTN_BLOCK_M` | `0` | Shared: overrides the query-block size (BLOCK_M). Must be a power of two and not smaller than queries per KV head. `0` is automatic. | leave default. | `vllm/envs.py:1591`; `vllm/v1/attention/ops/triton_unified_attention.py:1031-1040` |
| `VLLM_ROCM_ATTN_STAGES` | `0` | Shared: overrides the software-pipeline stage count. `0` means one stage. | leave default. | `vllm/envs.py:1894`; `vllm/v1/attention/ops/triton_unified_attention.py:1381-1396` |
| `VLLM_ROCM_ATTN_3D_SPEC` | `1` | Allows the 3D (split-K) attention kernel also when a sequence has several query tokens, that is under speculative decoding. With `0` the 2D grid path is used. | leave default; the working notes say MTP is about three times slower without it. | `vllm/envs.py:1900`; `vllm/v1/attention/backends/triton_attn.py:213`; `vllm/v1/attention/ops/triton_unified_attention.py:1291`; `vllm/v1/attention/ops/int4_per_token_head.py:935` |
| `VLLM_ROCM_SOFTMAX_SEGMENTS` | `0` | Number of softmax split segments for the 3D attention kernel on gfx10. `0` applies the context-length rule; a positive value fixes the count. | leave default; a fixed value only from a measured decode-occupancy sweep. | `vllm/envs.py:1529`; `vllm/v1/attention/backends/triton_attn.py:244-248` |

## 3. Quantized layer kernels (W4A16, W8A16, W4A8, lm_head, MoE, GDN)

| Variable | Default | What it does | Set it when | Evidence |
|---|---|---|---|---|
| `VLLM_ROCM_W4A16_TILE` | `""` | Manual tile sweep for the gfx10 W4A16 linear kernel, format `M,N,K,warps` with an optional stages field per the comment. Empty means automatic tiling. | leave default. | `vllm/envs.py:1595`; `vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py:351-352` |
| `VLLM_ROCM_W4A16_SPLIT_MAX` | `64` | Upper bound on token chunking of W4A16 linear calls. `0` disables the cap (A/B). | leave default. | `vllm/envs.py:1887`; `vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py:545` |
| `VLLM_ROCM_W4A16_DEQ_MIN` | `256` | Token count from which W4A16 switches to dequantize-then-GEMM (hipBLASLt). `0` disables. | leave default. | `vllm/envs.py:1883`; `vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py:612` |
| `VLLM_ROCM_SKINNY_EXT` | `1` | Uses the HIP extension `skinny_int4_ext` for W4A16 linears with M up to 8, and chunks of 8 up to `SPLIT_MAX`. If the extension does not build, the standard wvSplitK_int4_g path is kept. | leave default (set to 1 on 29.09.2026 by decision, per the envs comment). | `vllm/envs.py:1814`; `vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py:550` |
| `VLLM_ROCM_SMALLN_FIX` | `1` | Keeps small batches (6 to 16 rows) on fast kernels. The unquantized GEMM is split into wvSplitK chunks of at most 5 rows, and the int8 lm_head matvec takes up to 16 rows. | leave default. | `vllm/envs.py:1804`; `vllm/model_executor/layers/rdna2_matvec.py:145`; `vllm/model_executor/layers/utils.py:225` |
| `VLLM_ROCM_DISABLE_HYBRID_W4A16` | `0` | On gfx10, `1` rejects the RDNA hybrid W4A16 linear kernel, so vLLM uses the Triton W4A16 kernel. | leave default; `1` only for A/B. | `vllm/envs.py:1483`; `vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py:954-955` |
| `VLLM_ROCM_W4A8` | `0` | Token threshold for the W4A8 linear path (4-bit weights, signed 8-bit activations, integer dot product). Below the threshold the call uses W4A16. `0` disables. | leave default; set to the measured threshold (the published measurements used 256). | `vllm/envs.py:1787`; `vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py:737`, `:1148` |
| `VLLM_ROCM_W4A8_TILE` | `4` | Register-tile layout of the W4A8 HIP GEMM. `8` selects the 8x8 tile; any other value keeps the older 8x4 tile. The C++ reads the raw variable, so the envs default only documents the 8x4 path. | leave default; `8` where the measured prefill gain (about 10% on 27B, output byte-identical per the envs comment) is wanted. | `vllm/envs.py:1824`; `vllm/model_executor/kernels/linear/mixed_precision/w4a4/w4a4_ext.hip:785-786` |
| `VLLM_ROCM_W8A16` | `0` | Quantizes unquantized linear layers to int8 with a per-row scale at load time on gfx10. Decode uses the HIP `w8a16_skinny` kernel; prefill dequantizes and uses GEMM. The lm_head path also uses the HIP kernel. | leave default; `1` for the W8A16 configuration of the Qwen3.6 preset. | `vllm/envs.py:1799`; `vllm/model_executor/layers/linear.py:204`; `vllm/model_executor/layers/vocab_parallel_embedding.py:114` |
| `VLLM_ROCM_LMHEAD_INT8` | `0` | Quantizes lm_head to int8 per channel and computes logits in fp32 through rdna2_matvec. `1` frees the fp16 weight after quantization (not with tied embeddings); `2` quantizes but keeps the fp16 weight as a fallback. | leave default; `1` in measured configurations that need the memory and bandwidth saving. | `vllm/envs.py:1792`; `vllm/model_executor/layers/vocab_parallel_embedding.py:99`, `:778`, `:803-806`, `:846` |
| `VLLM_ROCM_MOE_RDNA2` | `0` | Selects the fork's native HIP fp16 W4A16 MoE kernel (gfx10) instead of the Triton MoE path, for compressed-tensors WNA16 MoE. Falls back if the device is not gfx10 or the extension is not built. Raises ValueError when weight scales are not fp16. | Set it when: running an MoE model on gfx103x in fp16 with the extension built. The published 35B-A3B measurements used `1`. | `vllm/envs.py:1835`; `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/rocm_moe_rdna.py:23`; `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py:93-96`; `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_rdna2.py:52` |
| `VLLM_ROCM_MOE_RDNA2_DETERMINISTIC` | `1` | With `VLLM_ROCM_MOE_RDNA2`, sums split-K partial results in fp32 in a fixed order instead of fp16 atomic adds. | leave default; `0` restores the atomic path, whose order varies between runs. | `vllm/envs.py:1840`; `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_rdna2.py:43` |
| `VLLM_ROCM_GDN_FAST_DECODE` | `0` | GatedDeltaNet decode fast path. `1` removes extra tensor copies and allocates the output with empty instead of zeros. `2` uses the fused HIP kernel (recurrent update plus RMSNormGated). | leave default; `1` or `2` only when decode speed is being benchmarked. | `vllm/envs.py:1626`; `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:615`, `:1040`, `:1080` |
| `VLLM_ROCM_GDN_DECODE_VIA_CHUNK` | `0` | Routes GatedDeltaNet decode through the blockwise kernel used for prefill, bypassing the packed step-by-step decode. | leave default; `1` where agreement with the blockwise path is wanted and slower decode is acceptable. | `vllm/envs.py:1622`; `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1424`, `:1517`, `:1710` |
| `VLLM_ROCM_ALLOW_FP32_GDN` | `0` | Removes the fp32 refusal in the GatedDeltaNet chunk kernel (RDNA2 has no bf16). Per the envs comment it also forces one Triton pipeline stage. | leave default; `1` only if GDN must run in fp32 on this card. | `vllm/envs.py:1638`; `vllm/third_party/flash_linear_attention/ops/chunk.py:223`; `vllm/third_party/flash_linear_attention/ops/utils.py:218` |
| `VLLM_GGUF_KN_MIN` | `8192` | unclear in fork: the reader is the external vllm-gguf-plugin, which is not in this fork. The envs comment says it sets the token count from which the second multiplier is written as [K, N]; `0` disables. | leave default (effect not verifiable in the fork). | `vllm/envs.py:1700` (no reader in the fork) |

## 4. Sampling and reasoning

These three change the model's output, not only its speed. See the section on model output below.

| Variable | Default | What it does | Set it when | Evidence |
|---|---|---|---|---|
| `VLLM_ROCM_MAX_TEMPERATURE` | `0` | When above 0, caps each request's temperature at this value, after argument validation. The cap is floored at the module constant `_MAX_TEMP`. `0` disables it. | leave default; set only with a quality measurement showing that the cap helps, since it changes sampling output. | `vllm/envs.py:1504`; `vllm/sampling_params.py:540` |
| `VLLM_ROCM_MIN_P_FLOOR` | `0` | When above 0, a request with positive temperature and no speculative config gets its `min_p` raised to at least this value. Greedy requests are not changed. | leave default; set only with a quality measurement, since it changes sampling output. The presets `qwen3.5-9b` and `qwen3.8-27b` set 0.05. | `vllm/envs.py:1512`; `vllm/sampling_params.py:837` |
| `VLLM_REASONEND_GRACE` | `0` | After the REASONEND hint, gives the model this many tokens to finish its reasoning by itself. `0` is the old hard cut-off. | leave default; a test used 2048 with a thinking budget of 4096. | `vllm/envs.py:1848`; `vllm/v1/sample/thinking_budget_state.py:83` |

## 5. Memory

| Variable | Default | What it does | Set it when | Evidence |
|---|---|---|---|---|
| `VLLM_ROCM_EMBED_UVA` | `0` | Keeps the embedding table in pinned host memory and maps it to the GPU through UVA. Any positive value enables it. The envs comment says this frees about 1.18 GiB per card. Skipped when embeddings are tied to lm_head. | leave default; use where VRAM is the limit and embeddings are not tied to lm_head. | `vllm/envs.py:1831`; `vllm/model_executor/layers/vocab_parallel_embedding.py:429`, `:612`, `:632` |
| `VLLM_ROCM_HEAD_CHUNK` | `0` | On ROCm, computes the fp32 lm_head in vocabulary bands of this width instead of one fp32 matmul. The comment reports a hipErrorIllegalAddress on the full head shape. `0` uses a single call. | leave default; set (the comment uses 8192) only when the fp32 head fails on the full vocabulary shape. | `vllm/envs.py:1879`; `vllm/model_executor/layers/logits_processor.py:155-158` |
| `VLLM_KV_PAGE_GROW_BLOCK` | `1` | When KV page sizes do not divide (Gemma 4 with int4 KV), enlarges the smaller layer's block by the integer part of the ratio and pads only the remainder. `0` pads by the whole ratio (upstream behaviour). | leave default; `0` only to reproduce the upstream KV layout. | `vllm/envs.py:1737`; `vllm/v1/core/kv_cache_utils.py:1095` |

## 6. Pipeline parallel

| Variable | Default | What it does | Set it when | Evidence |
|---|---|---|---|---|
| `VLLM_PP_MICROBATCH_SPLIT` | `None` (unset) | Caps decode requests per scheduler step to about num_decode / pp_size, leaving the rest for the next step so stages can overlap. Unset means enabled when PP>1; `0` disables. Applies only without async scheduling. | leave default (unset); `0` only for A/B against unsplit scheduling. | `vllm/envs.py:957`; `vllm/v1/core/sched/scheduler.py:304`, `:551-554` |
| `VLLM_PP_HOST_TRANSFER` | `0` | Sends pipeline-stage tensors through host memory and a gloo group instead of RCCL. Needed when a card sits behind a chipset root port without PCIe atomics, where RCCL 2.27+ kernels fail. Incompatible options are rejected at startup. | `1` for pipeline parallelism on cards without PCIe atomics (the 3-card configuration of the measurements); otherwise leave default. | `vllm/envs.py:967`; `vllm/distributed/parallel_state.py:2338`, `:2354` |
| `VLLM_PP_HOST_SEND_ASYNC` | `0` | With `VLLM_PP_HOST_TRANSFER`, copies stage output to host asynchronously, and a background thread waits for the pass and sends over gloo. Has no effect without HOST_TRANSFER and PP>1; a warning is logged. | `1` for the async-send configuration of the three-card measurements (requires `VLLM_PP_HOST_TRANSFER=1`); otherwise leave default. | `vllm/envs.py:983`; `vllm/distributed/parallel_state.py:2365`, `:2373`; `vllm/v1/worker/gpu_model_runner.py:841`; `vllm/v1/worker/gpu_worker.py:1104` |
| `VLLM_ROCM_NO_SIGKILL` | `1` | On shutdown of `vllm serve`, logs the surviving worker pids and does not SIGKILL them, because SIGKILL on a process with pending GPU faults can hang the driver (per the comment). Upstream behaviour is `0`. Data-parallel serving and the sweep benchmark keep their upstream kills; no preset uses them. | leave default; `0` only for upstream shutdown semantics. Leftover processes then need a manual stop. | `vllm/envs.py:980`; `vllm/v1/utils.py:490`, `:642`; `vllm/v1/executor/multiproc_executor.py:494` |

## 7. Diagnostics and probes

Measurement-only switches (`VLLM_ROCM_ABLATE`, `VLLM_ROCM_SKIP_ALL_REDUCE`) produce wrong model output by design.

| Variable | Default | What it does | Set it when | Evidence |
|---|---|---|---|---|
| `VLLM_ROCM_ABLATE` | `""` | Measurement only. A comma-separated list of parts to remove from the step: `gdn` (GatedDeltaNet kernels), `attn` (full attention), `lmhead` (vocabulary projection), `w4a16` (W4A16 linears). Removed parts return zeros, so the output is wrong. | measuring the cost of one component; never in serving. | `vllm/envs.py:1526`; `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1877`; `vllm/model_executor/layers/attention/attention.py:769`; `vllm/model_executor/layers/logits_processor.py:112`; `vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py:1101` |
| `VLLM_ROCM_SKIP_ALL_REDUCE` | `0` | Measurement only. Skips the TP all-reduce and returns the local tensor, so output is wrong and only the step cost remains visible. | measuring the cost of TP communication; never in serving. | `vllm/envs.py:1517`; `vllm/distributed/device_communicators/cuda_communicator.py:316-317` |
| `VLLM_ROCM_TRACE_ATTN` | `0` | Prints once which attention path was selected (3D or 2D, grid, tile). | verifying which attention kernel ran. | `vllm/envs.py:1891`; `vllm/v1/attention/ops/int4_per_token_head.py:964`; `vllm/v1/attention/ops/triton_unified_attention.py:1309` |
| `VLLM_ROCM_TRACE_LAYERS` | `""` | Path prefix. When set, the Qwen3-Next model appends per-layer state to `<prefix>.rank<N>`. Empty means off. | diagnosing per-layer state divergence between runs or ranks. | `vllm/envs.py:1651`; `vllm/model_executor/models/qwen3_next.py:598`, `:613` |
| `VLLM_ROCM_TRACE_SAMPLED` | `""` | Path prefix. When set, each TP rank appends its sampled token ids to `<prefix>.rank<N>` every step. Adds a GPU-to-CPU sync per step. | diagnosing cross-rank sampling divergence. | `vllm/envs.py:1648`; `vllm/v1/worker/gpu_model_runner.py:3998-4002` |
| `VLLM_ROCM_TIME_SPEC` | `0` | Per-phase timing report for the speculative draft step, printed every N steps and averaged over the last N calls. `0` is off. | profiling speculative decoding. | `vllm/envs.py:1597`; `vllm/v1/spec_decode/llm_base_proposer.py:78`, `:102` |
| `VLLM_PP_STAGE_TRACE` | `0` | Enables the per-stage pipeline timing recorder. Each rank writes `rank{N}.jsonl`, the engine writes `engine.jsonl` (one line per microbatch). Off adds no calls on the hot path, per the comment. | measuring pipeline stage timing. | `vllm/envs.py:990`; `vllm/v1/pp_stage_trace.py:5`; `vllm/v1/engine/core.py:221`; `vllm/v1/worker/gpu_worker.py:455` |
| `VLLM_PP_STAGE_TRACE_DIR` | `"/tmp/vllm-rdna2/pp-stage-trace"` | Output directory for the stage trace files. | leave default unless the location must change. | `vllm/envs.py:991`; `vllm/v1/pp_stage_trace.py:221`, `:227` |
| `VLLM_PROBE_OUTPUT_DIR` | `None` (unset) | Output directory for logit and hidden-state probes. dump_logits falls back to `/tmp/vllm-rdna2/probe`. probe_hooks raises KeyError when it is unset. | using the probes. | `vllm/envs.py:1678`; `vllm/v1/sample/logits_processor/dump_logits.py:55`; `vllm/v1/sample/logits_processor/probe_hooks.py:326-328` |
| `VLLM_PROBE_SAVE_PER_STEP` | `1` | dump_logits writes the logits of every step as a separate record. Accepts `1`, `true` or `yes`. | leave default. | `vllm/envs.py:1680`; `vllm/v1/sample/logits_processor/dump_logits.py:58` |
| `VLLM_PROBE_TOP_K` | `5` | Number of most likely tokens written to the dump_logits report. | leave default. | `vllm/envs.py:1684`; `vllm/v1/sample/logits_processor/dump_logits.py:59` |
| `VLLM_PROBE_PROMPT_TEXT` | `""` | Prompt text placed in the dump_logits report header. | leave default. | `vllm/envs.py:1686`; `vllm/v1/sample/logits_processor/dump_logits.py:64` |
| `VLLM_PROBE_LAYERS` | `"17"` | Comma-separated layer numbers where probe hooks are attached. | leave default unless other layers are probed. | `vllm/envs.py:1688`; `vllm/v1/sample/logits_processor/probe_hooks.py:324` |
| `VLLM_PROBE_HOOK_PATTERN` | `None` (unset) | Regular expression matched against module names for probe hooks. Unset or empty means no pattern. | leave default. | `vllm/envs.py:1690`; `vllm/v1/sample/logits_processor/probe_hooks.py:325` |
| `VLLM_PROBE_ALL_LAYERS` | `0` | Saves all layers instead of only those in `VLLM_PROBE_LAYERS`. Enabled only when the value is `"1"`. | leave default. | `vllm/envs.py:1692`; `vllm/v1/sample/logits_processor/probe_hooks.py:332` |

## 8. Experimental, not recommended

| Variable | Default | What it does | Set it when | Evidence |
|---|---|---|---|---|
| `VLLM_ROCM_W4A4` | `0` | Token threshold for the W4A4 linear path (4-bit weights and 4-bit activations, integer dot product). Below it the normal path is used. Changes numerical results, not only speed. | leave default; not a default until quality is measured. | `vllm/envs.py:1763`; `vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py:696`, `:1058`, `:1173` |
| `VLLM_ROCM_W4A4_SIM` | `0` | Quality-analysis switch. Simulates 4-bit activation quantization in PyTorch, then runs the normal W4A16 path. Modes: `1` per row; `2` per 128-group; `3` per group, non-RowParallel layers only; `4` per group, RowParallel only; `5` Hadamard rotation within the group plus per-group, all layers; `6` as 5 but non-RowParallel only. | quality analysis only.  | `vllm/envs.py:1770`; `vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py:869-905` (docstring and implementation), `:1087`, `:1132-1138` |
| `VLLM_ROCM_W4A4_SKIP_ROW` | `0` | Excludes RowParallelLinear layers (down_proj, o_proj) from the W4A4 path. | leave default; `>0` only to keep those two layers in W4A16 during analysis. | `vllm/envs.py:1774`; `vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py:1182` |
| `VLLM_ROCM_W4A4_ROT` | `0` | At load time, rotates weights with a Hadamard transform within 128-groups and requantizes them; activations are rotated before the matmul. Per the envs comment, it needs AWQ scale refitting. | leave default; experimental. | `vllm/envs.py:1781`; `vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py:1035-1058` |
| `VLLM_ROCM_FP32_RESIDUAL` | `0` | Accumulates the residual stream in fp32 instead of the model's fp16 in the Qwen3-Next path. Weights, multiplications and layers stay fp16. | leave default; `1` only to test a quality hypothesis, with a quality check. | `vllm/envs.py:1615`; `vllm/model_executor/models/qwen3_next.py:401-416` (used at `:524`, `:528`, `:568`, `:738`) |
| `VLLM_ALLOW_PP_SPEC` | `0` | Overrides the refusal to run speculative decoding with pipeline parallelism. The envs comment says outputs are corrupted in that combination. | debugging only; never for serving. | `vllm/envs.py:1672`; `vllm/config/vllm.py:1631`; `vllm/v1/spec_decode/llm_base_proposer.py:1622` |
| `VLLM_DYN_SD_FULL_GRAPHS` | `0` | With dynamic speculative decoding under MRV1, captures full CUDA graphs per query length (1 + K). With `0` the cudagraph mode is forced to PIECEWISE. | leave default; `1` only for a measured dynamic-SD graph experiment. | `vllm/envs.py:1845`; `vllm/config/vllm.py:900`, `:919`; `vllm/v1/cudagraph_dispatcher.py:40` (comment) |
| `VLLM_GGUF_MMQ_IQ3S` | `0` | unclear in fork: the reader is the external vllm-gguf-plugin. The envs comment says it selects the MMQ path for IQ3_S and is off because measurement showed it unfavourable. | leave default. | `vllm/envs.py:1702` (no reader in the fork) |
| `VLLM_GGUF_SHARD_CACHE` | `0` | unclear in fork: the reader is the external vllm-gguf-plugin. The envs comment says it caches contiguous shards with mixed types; it is also part of the torch.compile cache key. | leave default. | `vllm/envs.py:1704` (no reader in the fork) |
| `VLLM_ROCM_INT4_EDGE_MASK` | `0` | Skips the sequence-mask computation for tiles entirely inside the causal triangle in the int4 kernel. Off by default: the comment reports +4.9% kernel time alone and +56% with int8 P·V, with the mechanism unexplained. | leave default. | `vllm/envs.py:1556`; `vllm/v1/attention/ops/int4_per_token_head.py:1156` |
| `VLLM_ROCM_FLA_SINGLE_CONFIG` | `0` | Uses one autotuner configuration for the FLA block kernels regardless of head count (experiment of 29.08.2026). | leave default. | `vllm/envs.py:1632`; `vllm/third_party/flash_linear_attention/ops/utils.py:210` |

## Switches that change the model output, not only the speed

Each entry is verified from code or from a code comment. Perf-only switches are omitted.

- `VLLM_ROCM_MAX_TEMPERATURE`: caps request temperature (`vllm/sampling_params.py:540`).
- `VLLM_ROCM_MIN_P_FLOOR`: raises min_p (`vllm/sampling_params.py:837`).
- `VLLM_REASONEND_GRACE`: changes how thinking ends (`vllm/v1/sample/thinking_budget_state.py:83`).
- `VLLM_ROCM_TP_SYNC_SAMPLED` and `VLLM_ROCM_TP_SYNC_VIA_ALLREDUCE`: decide whether TP ranks agree on the token; output differs between on and off (`vllm/v1/worker/gpu_model_runner.py:3907-3925`).
- `VLLM_ROCM_GDN_DECODE_VIA_CHUNK`: per the envs comment, the packed decode differs from the blockwise path on 10.6% of positions (envs.py comment above line 1622).
- `VLLM_ROCM_W4A4`, `VLLM_ROCM_W4A4_SIM`, `VLLM_ROCM_W4A4_ROT`, `VLLM_ROCM_W4A4_SKIP_ROW`: 4-bit activations, numerical change.
- `VLLM_ROCM_W4A8`, `VLLM_ROCM_INT4_W4A8`, `VLLM_ROCM_INT4_QBITS` (values 6 and 4), `VLLM_ROCM_INT4_PV_INT8`, `VLLM_ROCM_INT8_PV_INT8`, `VLLM_ROCM_QK_INT8`, `VLLM_ROCM_INT4_FP16_DOT`: quantized activations, queries or probabilities, or fp16 accumulation.
- `VLLM_ROCM_INT4_RHT=0`: quality collapse, see `docs/notes/measurements/2026-09-29-gemma4-26b-a4b.md`.
- `VLLM_ROCM_W8A16`, `VLLM_ROCM_LMHEAD_INT8`: int8 weights.
- `VLLM_ROCM_FP32_RESIDUAL`, `VLLM_ROCM_FP32_ALLREDUCE`: rounding of residual and reduction sums.
- `VLLM_ROCM_MOE_RDNA2`: a different kernel. Its numerical equality with the Triton MoE path is not verified.
- `VLLM_ROCM_MOE_RDNA2_DETERMINISTIC`: fixes the summation order, so results are bit-different from the atomic path.
- `VLLM_ROCM_GDN_FAST_DECODE=2`: fused kernel. Numerical equality is not verified.
- `VLLM_ROCM_GEMMA4_FUSE_FFN_AR`: fused reduce order, not bit-identical to upstream at TP of 3 or more, per the comment.
- `VLLM_ROCM_ABLATE`, `VLLM_ROCM_SKIP_ALL_REDUCE`: wrong output by design (measurement only).
- `VLLM_ALLOW_PP_SPEC=1`: the envs comment says output is corrupted.
- `VLLM_GGUF_*`: unclear in fork (external reader).

Non-environment switch, listed here because it changes output: `anti_stutter` is a logits processor enabled by the command-line option `--logits-processors vllm.v1.sample.logits_processor.anti_stutter:AntiStutter` (`vllm/v1/sample/logits_processor/anti_stutter.py:28`). It is not controlled by any environment variable. It forbids an immediate repeat of a subword pair. Its own docstring (lines 1-22) warns that its metric falls by construction and does not prove the text is better.

## Not settled

- The effect of the three `VLLM_GGUF_*` switches: they are read by an external plugin that is not part of this release.
- Whether `VLLM_ROCM_MOE_RDNA2` and `VLLM_ROCM_GDN_FAST_DECODE=2` reproduce the results of the unfused paths bit for bit.
- Whether the native broadcast of sampled tokens is reliable on this pair: see KI-1 in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).
