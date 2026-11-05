# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/). Before 1.0.0, presets, scripts and switches can change between minor versions.

## 0.1.0 - 2026-11-05

First public release. A patch against vLLM `c296851a7d`, with an installer, presets for five models, and all measurements.

### Added

**Tensor parallelism without P2P**

- All-reduce through shared host memory for two ranks: 17 µs against about 164 µs through RCCL at the decode size in a microbenchmark, and about +7 % decode with one stream on Qwen3.8-27B and Qwen3.5-9B. It also works inside captured CUDA graphs (`VLLM_ROCM_HOST_ALL_REDUCE`, on by default).
- `all_gather` and `reduce_scatter` built from all-reduce, because the RCCL versions return wrong data on this pair (`VLLM_ROCM_COLLECTIVES_VIA_ALLREDUCE`, on by default).
- Agreement of sampled tokens between TP ranks (`VLLM_ROCM_TP_SYNC_SAMPLED`).

**Platform**

- Detection of gfx10 in `vllm/platforms/rocm.py`, with tile sizes tuned for gfx1030.

**Kernels**

- W4A8 and W4A4 linear layers through `v_dot4` and `v_dot8`, W8A16 for unquantized layers, a matrix-vector kernel for `lm_head`.
- W4A16 kernel for up to 8 rows, which removed the loss of MTP with two streams (`skinny_int4_ext`, `VLLM_ROCM_SKINNY_EXT`, on by default).
- Native W4A16 kernel for MoE experts, with reproducible results by default (`VLLM_ROCM_MOE_RDNA2`).
- int4 KV cache with one scale per token and head, HIP decode attention and HIP prefill attention.
- Fused decode for Gated DeltaNet layers (`VLLM_ROCM_GDN_FAST_DECODE`).

**Operation**

- `scripts/kill-stuck-rank.sh --force` and `serve.sh --kill-stuck`: SIGKILL for a rank that survives SIGTERM, only if the kernel log can be read and has no page fault from it or on its card. Off by default.

**Three cards**

- Pipeline parallelism through host memory for cards without PCIe AtomicOps (`VLLM_PP_HOST_TRANSFER`).

**Tools**

- `install.sh`: installs at pinned versions and never uses `sudo`.
- `serve.sh`, `stop.sh`: start with retries, a driver check and a first test request; stop without SIGKILL.
- `scripts/`: `doctor.sh`, `gpu-health.sh`, `kill-stuck-rank.sh`, `download-model.sh`, `add-mtp-head.py`, `awq-to-ct.py`, `install-service.sh`, `smoke-test.py`.
- Presets for Qwen3.8-27B, Qwen3.5-9B, Qwen3.5-35B-A3B, Qwen3.6-35B-A3B and Gemma 4 26B-A4B.
- `bench/`: throughput matrix, all-reduce latency and correctness, long-context retrieval.
- `results/`: the measurement records and the generator of every table in the documentation.
- `tests/`: tests of the scripts and tables, with no GPUs needed.

### Fixed against upstream vLLM on this target

- MTP on hybrid models with TP=2: ranks accepted different draft counts under random sampling, which broke the recurrent state.
- `ROCM_ATTN` and the generic kernel branches being selected, because gfx10 belongs to no GPU family in the platform code.
- vLLM's skinny GEMM kernels return wrong results on gfx1030. They stay disabled for gfx10; the cause is not known.
- SIGKILL of workers at every stop (`VLLM_ROCM_NO_SIGKILL`, on by default).

### Known limits

See [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md). The main ones: random faults at start or on the first request, a hang on a client abort during prefill with MTP, and no authentication in the server.
