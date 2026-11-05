# Working notes (Ukrainian, unedited)

These are the working notes of the project, kept in Ukrainian. The only changes are
privacy substitutions, the removal of sections about the author's local client
setup and private disks, links into a local checkout reduced to plain text, and one
dated correction in `measurements/perf-numbers.md`. Host names, network addresses,
home paths and local tool names are replaced with placeholders, and each removed
section leaves a one-line placeholder in its place.

The current documentation is in [`docs/`](../); these notes keep the history of
the measurements and decisions behind it. Links between the notes resolve. Paths
such as `work/...` and `tests/tools/...` belong to the working repository, which is
not public.

The notes were written during development with AI coding assistants: «підагент»
(subagent) and names such as `agy`, Sonnet or Claude refer to them. A claim that a
note attributes to an assistant is not a measurement.

## Index

Top level:

- `gdn-decode-launches.md`: kernel launches per decode step in the GDN layers.
- `int4-gemm-hypotheses.md`: int4 GEMM on gfx1030, list of hypotheses and their status.
- `int4-speed-timeline.md`: speed of the int4 path over two days.
- `machine-config.md`: server configuration of the test machine after migration.
- `pipeline-int4-w4a8.md`: where time goes in the int4/W4A8 pipeline.
- `plan-w4a4.md`: W4A4 on gfx1030: state and next steps.
- `rdna2-isa-notes.md`: RDNA 2 (gfx1030) ISA notes for matrix multiplication.
- `report-w4a4-w4a8.md`: W4A4 and W4A8, full report of the end-to-end measurements.
- `vllm-gfx1030.md`: vLLM on two Navi 21 cards (gfx1030), main working log.
- `w4a16-paths-map.md`: map of the W4A16 linear-layer multiplication paths.

Measurements (`measurements/`):

- `2026-09-19-concurrency-and-mtp-speed.md`: concurrency, two cards and MTP at long context.
- `2026-09-23-llamacpp-vs-vllm-3cards.md`: llama.cpp against vLLM on three cards.
- `2026-09-23-pp3-three-cards.md`: pipeline parallel size 3 on three cards.
- `2026-09-27-decode-scaling-results.md`: decode scaling results.
- `2026-09-27-decode-continuation.md`: the exact pair of decode configurations and its quality.
- `2026-09-27-layer-overhead.md`: small operations per layer.
- `2026-09-27-pp-capacity.md`: capacity and long prompts with 4 and 6 streams.
- `2026-09-27-pp-final-matrix.md`: full matrix of pipeline parallelism.
- `2026-09-27-pp-step-cost.md`: parts of a pipeline-parallel step and the limit of the model.
- `2026-09-27-skinny-study.md`: the skinny kernel, base and search.
- `2026-09-28-mtp-attention-results.md`: MTP and the HIP attention kernel at long context.
- `2026-09-29-night-results.md`: MTP with several streams, the kernel for up to 8 rows (`skinny_int4_ext`) and the HIP attention kernel for three query tokens.
- `2026-09-29-gemma4-26b-a4b.md`: Gemma 4 26B-A4B on two gfx1030 cards.
- `2026-09-29-moe-35b-a3b-rdna2.md`: Qwen3.5-35B-A3B (MoE) with the native HIP MoE kernel.
- `2026-10-03-qwen36-35b-a3b.md`: Qwen3.6-35B-A3B on two Navi 21 cards, first bring-up.
- `2026-10-07-qwen36-config-results.md`: Qwen3.6-35B-A3B configuration for an agent session.
- `known-defects.md`: list of known defects and open questions.
- `mtp-notes.md`: MTP mechanics, limits and defects.
- `offload-kv.md`: KV cache offload to system RAM, summary.
- `perf-numbers.md`: measured speeds.

Settings and numbers in these files describe the machine they were measured on.
Do not apply them to another machine without measuring it there.
