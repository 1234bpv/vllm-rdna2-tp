# Performance

All tables between the `table:` markers are generated from the records in `results/` by `results/make_tables.py`. Numbers in
the text cite their source file. No number here comes from another machine.

## How the numbers were measured

**Machine.** RX 6800 (60 CU) and RX 6800 XT (72 CU), 16 GB each, no P2P; 2 × Xeon E5-2699 v3; 128 GB DDR4-2133; GPU clock
limited to 1600 MHz; Kubuntu 24.04, Linux 6.8, ROCm 7.2.4, Python 3.12, PyTorch 2.13.0+rocm7.2, Triton 3.7.1, RCCL 2.27.7 from
the wheel. Base: vLLM `c296851a7d` (18 Aug 2026). llama.cpp `058df67`.

**When.** Unless stated otherwise, on 5 Nov 2026 on the release tree, installed from zero by `install.sh` into a fresh virtual
environment. A cell marked `*` holds a point that was not repeated on that day: its source is the run named in
[the configuration table](#configuration-of-each-run) (field `older` in `results/runs.json`). A few release points come from another start of the same day:
Qwen3.6-35B-A3B 32k × 2 and Qwen3.5-9B 300 × 2 and 8k × 4 from the A/B runs, Qwen3.6-35B-A3B + MTP 32k × 2 from the second start. The tables of KV cache formats, of the
effect of each kernel and of scaling compare runs of one earlier day, and their dates are in the configuration table.

**Probe.** `bench/kv-concurrency.py`: prompts of random tokens with a different seed per stream, 256 output tokens,
`temperature=0`, `ignore_eos`, token counts from the `usage` field of the server.

| Metric | Definition |
|---|---|
| PP | Prompt processing, tokens/s: the sum of the prompt tokens of all streams divided by the time until the last stream received its first token |
| TG | Decode, tokens/s: total output tokens divided by the shared window when all streams generate |
| TTFT | Seconds until the last stream received its first token |
| Prompt × streams | Prompt length in tokens (`32k` is 32,768; `125k` is 125,000) and the number of concurrent streams |
| KV cache, tokens | The line `GPU KV cache size` in the server log |

**In this repository "PP" always means prompt processing.** Pipeline parallelism is written out.

## Reading these numbers

1. **MTP numbers are inflated on random tokens.** Random prompts make the model fall into repetition, and most drafts are accepted (0.90–0.97
   at 300 tokens on Qwen3.5-9B).
   Compare MTP numbers only within one table. On a real workload (agent task set, 18 tasks, 4 clients, Qwen3.6-35B-A3B) MTP
   reduced the wall time by 5.6 %, from 189.8 to 179.1 s, with 73–77 % of the drafts accepted
   (`docs/notes/measurements/2026-10-07-qwen36-config-results.md`).
2. **Measurement error.** See [Repeats](#repeats). Without MTP, decode repeats within 3.8 %, and PP on long prompts within 5.1 %, so
   differences below about 5 % are noise. With MTP, two starts of the release tree differ in TG by up to 24.7 % at one point.
3. **PP of 300-token prompts with several streams is not shown** in the matrices: it reflects warm-up and repeats differ by up
   to 74.8 %.
4. **Cells marked `*` come from earlier code,** see [Configuration of each run](#configuration-of-each-run). The earlier Qwen3.5-35B-A3B
   records predate the HIP prefill kernel: on the release tree its PP at 125k × 1 is 1361 against 795, with the same TG.
5. **The KV cache size changes between starts** of the same preset: 584,782 to 712,493 tokens on Qwen3.5-35B-A3B, 268,101 and 205,544
   with MTP, 580,663 and 617,306 on Qwen3.8-27B, 742,741 to 867,091 on Qwen3.6-35B-A3B. The tables show the start of the main run. The cause is not established (KI-39).

## Summary

<!-- table:summary -->
| Model | Weights | Context | KV cache, tokens | PP 300 | PP 32k | PP 125k | TTFT 125k, s | TG 300 | TG 32k | TG 125k | ΣTG 300 × 4 | ΣTG 125k × 2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B | W4A16 AWQ g128 | 262,144 | 580,663 | 630 | 555 | 283 | 441 | 38.0 | 36.7 | 34.6 | 116.7 | 57.2 |
| Qwen3.8-27B + MTP | W4A16 AWQ g128 | 262,144 | 378,352 | 610 | 533 | 271 | 461 | 73.0 | 66.0 | 63.5 | 103.6 | 81.8 |
| Qwen3.5-9B | W4A16 AWQ g128 | 262,144 | 2,464,936 | 1676 | 2525 (8k) | — | — | 80.4 | 79.9 (8k) | — | 257.5 | — |
| Qwen3.5-9B + MTP | W4A16 AWQ g128 | 262,144 | 1,904,288 | 1500 | 2267 (8k) | — | — | 127.6 | 118.4 (8k) | — | 254.2 | — |
| Qwen3.5-35B-A3B | MoE W4A16 AWQ | 131,072 | 584,782 | 1181 | 2176 | 1361 | 92 | 52.8 | 52.1 | 50.0 | 175.2 | 80.8 |
| Qwen3.5-35B-A3B + MTP | MoE W4A16 AWQ | 131,072 | 268,101 | 1163 | 2036 | 1232 | 101 | 105.0 | 104.4 | 97.9 | 178.0 | 133.6 |
| Qwen3.6-35B-A3B | MoE CT-INT4 + W8A16 | 131,072 | 860,370 | 1154 | 2094 | 1289 | 97 | 66.1 | 64.2 | 58.8 | 206.3 | 99.8 |
| Qwen3.6-35B-A3B + MTP | MoE CT-INT4 + W8A16 | 131,072 | 542,161 | 993 | 1919 | 1212 | 103 | 96.2 | 124.6 | 115.9 | 236.1 | 136.5 |
| Qwen3.6-35B-A3B + MTP | MoE CT-INT4 + W8A16 | 262,144 | 655,360 | 668 | 1933 | 1158 | 108 | 104.1 | 123.7 | 102.3 | 166.1 | 149.0 |
| Gemma 4 26B-A4B | MoE AWQ INT4 g32 | 262,144 | 933,930 | 211 | 1733 | 1084 | 115 | 60.6 | 56.6 | 49.7 | 187.1 | 86.3 |
<!-- /table:summary -->

`(8k)` marks a prompt of 8,192 tokens where the 32k point does not exist for that run. `—` means not measured.

## All-reduce

Exchange time on the tested pair, measured in a loop (a microbenchmark) on the release tree on 5 Nov 2026: `bench/hostar_test.py` for this fork and `bench/arlat.py`
for RCCL, medians of 6 runs without pinning. At 5,120 elements the range is 16.1–18.6 µs against 127–177 µs. Five more runs pinned to one socket gave
15.9–21.3 µs and 125–185 µs.

<!-- table:all-reduce -->
| fp16 elements | Size | This fork, µs | RCCL, µs | RCCL / fork |
|---|---:|---:|---:|---:|
| 5,120 (decode exchange) | 10 KiB | 17.1 | ~164 | ×9.6 |
| 10,240 | 20 KiB | 22.6 | ~128 | ×5.6 |
| 40,960 | 80 KiB | 64.5 | ~128 | ×2.0 |
<!-- /table:all-reduce -->

An earlier record, from August 2026, showed 10.3, 11.7, 26.3 and 141.9 µs for this fork at 5,120, 10,240, 40,960 and 262,144 elements, against about 150, 120, 133
and 150 µs for RCCL. The RCCL values were reproduced, and the values of this fork were not: the cause is not established.

For comparison, the RCCL all-reduce on large messages tops out at about 5.7 GiB/s on this machine, while its PCIe 3.0 x16 link to the
CPU carries about 15.75 GB/s. This is the cost of copying twice through the host (`docs/notes/vllm-gfx1030.md`).

**The cost of all exchanges in a decode step,** August 2026, an earlier tree. Measured by removing them (`VLLM_ROCM_SKIP_ALL_REDUCE=1`), which gives wrong model
output but shows the time:

<!-- table:all-reduce-cost -->
| Decode of Qwen3.8-27B, one stream | TG, tokens/s | Step, ms |
|---|---:|---:|
| With all exchanges | 18.56 | 54.0 |
| No exchange at all (wrong output, timing only) | 20.16 | 49.7 |
| Cost of all exchanges | — | 4.3 (8 %) |
<!-- /table:all-reduce-cost -->

The PyTorch profiler had attributed 49.4 % of GPU time to `ncclDevKernel`, 130 calls per token at 414 µs each. Removing the exchange,
with the host all-reduce in place, showed 8 %. The profiler records a spin-waiting kernel as busy while the other card does the useful work, and its absolute
numbers were about four times too large: a 199 ms step against 54 ms in reality.

**The decode ladder of the day when the kernel was introduced.** Qwen3.8-27B, TP=2, one stream, 1600 MHz:

<!-- table:decode-chain -->
| Change, applied in this order | TG, tokens/s | Step gain |
|---|---:|---:|
| Start of the day: `--enforce-eager` | 5.06 | — |
| Captured CUDA graphs instead of eager mode | 13.70 | ×2.71 |
| All-reduce through host memory | 14.85 | ×1.08 |
| Own matrix-vector kernel for `lm_head` | 18.56 | ×1.25 |
| The same kernel for the small GDN projections | 18.93 | ×1.02 |
<!-- /table:decode-chain -->

**Direct A/B on the release tree.** The same points with `VLLM_ROCM_HOST_ALL_REDUCE=0` (RCCL) and `=1` (default), where the all-reduce is the only difference between
the runs. Qwen3.8-27B has three starts of each, the others one.

<!-- table:all-reduce-ab -->
| Model | Prompt × streams | TG, RCCL | TG, this fork | TG gain | PP, RCCL | PP, this fork | Runs |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B | 300 × 1 | 35.3 | 37.7 | +6.9 % | — | — | 3 / 3 |
| Qwen3.8-27B | 32k × 1 | 34.1 | 36.8 | +7.9 % | 555 | 556 | 3 / 3 |
| Qwen3.5-9B | 300 × 1 | 74.9 | 79.9 | +6.6 % | — | — | 1 / 1 |
| Qwen3.5-9B | 300 × 4 | 258.5 | 254.5 | -1.6 % | — | — | 1 / 1 |
| Qwen3.5-9B | 8k × 1 | 74.1 | 79.1 | +6.8 % | 2524 | 2537 | 1 / 1 |
| Qwen3.5-9B | 8k × 4 | 248.3 | 246.6 | -0.7 % | 2534 | 2555 | 1 / 1 |
| Qwen3.6-35B-A3B | 300 × 1 | 67.4 | 65.4 | -2.9 % | — | — | 1 / 1 |
| Qwen3.6-35B-A3B | 300 × 4 | 206.4 | 204.6 | -0.9 % | — | — | 1 / 1 |
| Qwen3.6-35B-A3B | 32k × 1 | 65.1 | 63.7 | -2.1 % | 2083 | 2138 | 1 / 1 |
| Qwen3.6-35B-A3B | 32k × 2 | 110.6 | 113.7 | +2.7 % | 2135 | 2167 | 1 / 1 |
<!-- /table:all-reduce-ab -->

With one stream, decode gains 6.6–7.9 % on Qwen3.8-27B and Qwen3.5-9B. With several streams and on Qwen3.6-35B-A3B the differences are between −2.9 % and +2.7 %,
which is within the spread of repeats, so no gain is shown there. Why the gain disappears has not been investigated. PP differs by at most 2.6 %. The three starts
of each mode on Qwen3.8-27B repeat within 1.1 %. The ladder above shows ×1.08 for the day when the kernel was introduced.

## Full matrices

Each cell is **PP / TG** in tokens/s.

### Qwen3.8-27B

<!-- table:matrix-q27 -->
| Prompt × streams | No MTP, PP / TG | MTP 2, PP / TG |
|---|---:|---:|
| 300 × 1 | 630 / 38.0 | 610 / 73.0 |
| 300 × 2 | — / 69.9 * | — / 103.0 * |
| 300 × 4 | — / 116.7 | — / 103.6 |
| 32k × 1 | 555 / 36.7 | 533 / 66.0 |
| 32k × 2 | 572 / 65.0 * | 551 / 101.9 * |
| 32k × 4 | 577 / 105.5 * | 551 / 111.7 * |
| 125k × 1 | 283 / 34.6 | 271 / 63.5 |
| 125k × 2 | 281 / 57.2 | 256 / 81.8 |
<!-- /table:matrix-q27 -->

### Qwen3.5-9B

<!-- table:matrix-q9 -->
| Prompt × streams | No MTP, PP / TG | MTP 2, PP / TG |
|---|---:|---:|
| 300 × 1 | 1676 / 80.4 | 1500 / 127.6 |
| 300 × 2 | — / 150.6 | — / 194.6 |
| 300 × 4 | — / 257.5 | — / 254.2 |
| 8k × 1 | 2525 / 79.9 | 2267 / 118.4 |
| 8k × 4 | 2555 / 246.6 | 2063 / 244.3 |
<!-- /table:matrix-q9 -->

### Qwen3.5-35B-A3B

<!-- table:matrix-q35 -->
| Prompt × streams | No MTP, PP / TG | No MTP, KV offload 32 GiB, 1 Oct, PP / TG | MTP 2, PP / TG |
|---|---:|---:|---:|
| 300 × 1 | 1181 / 52.8 | — | 1163 / 105.0 |
| 300 × 2 | — / 97.0 | — | — / 151.4 |
| 300 × 4 | — / 175.2 | — | — / 178.0 |
| 32k × 1 | 2176 / 52.1 | — | 2036 / 104.4 |
| 32k × 2 | 2192 / 92.3 | — | 2102 / 157.0 |
| 32k × 4 | 2129 / 164.0 | — | 1646 / 190.4 * |
| 125k × 1 | 1361 / 50.0 | 778 / 51.2 | 1232 / 97.9 |
| 125k × 2 | 1319 / 80.8 | — | 1266 / 133.6 |
| 125k × 4 | — | 840 / 144.8 | — |
<!-- /table:matrix-q35 -->

The point 32k × 4 with MTP did not fit in the start of 5 Nov 2026: that start had a KV cache of 205,544 tokens, and the streams were preempted 8 times, so the record does not
count. The cell is from 1 Oct, with a KV cache of 327,680 tokens. The point 125k × 4 was measured only without MTP and with KV offload to host memory. With MTP the head and its graphs take VRAM,
and the four long prompts did not fit together (`docs/notes/measurements/2026-09-29-moe-35b-a3b-rdna2.md`).

### Qwen3.6-35B-A3B

<!-- table:matrix-q36 -->
| Prompt × streams | No MTP, 131k, PP / TG | MTP 2, 131k, 7 Oct, PP / TG | MTP 2, 262k, PP / TG | MTP 2, 262k, int8 KV, 7 Oct, PP / TG |
|---|---:|---:|---:|---:|
| 300 × 1 | 1154 / 66.1 | 993 / 96.2 | 668 / 104.1 | 1087 / 105.4 |
| 300 × 4 | — / 206.3 | — / 236.1 | — / 166.1 | — / 200.0 |
| 32k × 1 | 2094 / 64.2 | 1919 / 124.6 | 1933 / 123.7 | 1401 / 105.8 |
| 32k × 2 | 2167 / 113.7 | 2024 / 167.8 | 2066 / 167.2 | 1478 / 123.6 |
| 125k × 1 | 1289 / 58.8 | 1212 / 115.9 | 1158 / 102.3 | 547 / 52.2 |
| 125k × 2 | 1348 / 99.8 | 1262 / 136.5 | 1241 / 149.0 | 607 / 43.6 |
<!-- /table:matrix-q36 -->

### Gemma 4 26B-A4B

<!-- table:matrix-gemma -->
| Prompt × streams | HIP prefill kernel, PP / TG | Before the HIP prefill kernel, PP / TG |
|---|---:|---:|
| 300 × 1 | 211 / 60.6 | — |
| 300 × 4 | — / 187.1 | — |
| 32k × 1 | 1733 / 56.6 | — |
| 64k × 8 | — | 858 / 198.7 |
| 125k × 1 | 1084 / 49.7 | — |
| 125k × 2 | 1123 / 86.3 | — |
| 125k × 4 | — | 555 / 135.5 |
| 250k × 1 | 692 / 42.8 * | 306 / 41.7 |
<!-- /table:matrix-gemma -->

The PP of 211 at 300 × 1 is the first point after the start (TTFT 1.4 s), so it reflects warm-up.

## Against llama.cpp

### Qwen3.6-35B-A3B on the same two cards

Fork: TP=2, CT-INT4 + W8A16, int4 KV cache, no MTP, window 131,072. llama.cpp `058df67`: Q5_K_M, q8 KV cache, no MTP, window 262,144.
Same probe, same day, 5 Nov 2026.

<!-- table:vs-llamacpp -->
| Prompt × streams | Fork PP | Fork TG | Fork TTFT, s | llama.cpp PP | llama.cpp TG | llama.cpp TTFT, s | PP ratio | TG ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 × 1 | — | 66.1 | 0.3 | — | 58.9 | 0.6 | — | ×1.12 |
| 300 × 4 | — | 206.3 | 0.8 | — | 132.2 | 1.3 | — | ×1.56 |
| 32k × 1 | 2094 | 64.2 | 16 | 1317 | 36.3 | 25 | ×1.59 | ×1.77 |
| 32k × 2 | 2167 | 113.7 | 30 | 946 | 60.5 | 69 | ×2.29 | ×1.88 |
| 125k × 1 | 1289 | 58.8 | 97 | 804 | 19.9 | 156 | ×1.60 | ×2.95 |
| 125k × 2 | 1348 | 99.8 | 185 | 463 | 27.3 | 540 | ×2.91 | ×3.65 |
<!-- /table:vs-llamacpp -->

Conditions that favor one side or the other:

- **The quantization differs, and the GGUF has the better quality**, see [Quality](#quality).
- **The windows differ:** 131,072 in the fork, 262,144 in llama.cpp. With MTP, the fork measured the same speed on both windows
  within 8 % on 7 Oct 2026, before the larger spread of MTP was seen ([matrix](#qwen36-35b-a3b)).
- **llama.cpp with MTP did not start at 262,144:** the draft KV cache ran out of memory (512 MiB), so this comparison has no MTP on
  either side. The 131,072 repeat was cancelled. An older record of Qwen3.8-27B in llama.cpp below shows what MTP gave there.
- **PP of 300-token prompts is not compared:** it reflects warm-up.
- **The first requests to llama.cpp are slower.** At 300 × 1 the first two requests after its start gave a TG of 48.1 and 51.3, and the table shows the third, 58.9.

### Qwen3.8-27B: two cards against three

Fork: TP=2 on two cards, int4 KV cache. llama.cpp: layer split on three cards (the third on a chipset PCIe 2.0 x1 slot), Q4_0, q4_0 KV
cache. No MTP on either side. Same probe, same day, 5 Nov 2026.

<!-- table:vs-llamacpp-q27 -->
| Prompt × streams | Fork PP | Fork TG | Fork TTFT, s | llama.cpp PP | llama.cpp TG | llama.cpp TTFT, s | PP ratio | TG ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 × 1 | — | 38.0 | 0.5 | — | 20.4 | 2.0 | — | ×1.86 |
| 32k × 1 | 555 | 36.7 | 59 | 369 | 13.5 | 89 | ×1.50 | ×2.71 |
| 125k × 1 | 283 | 34.6 | 441 | 231 | 6.5 | 540 | ×1.22 | ×5.31 |
| 125k × 2 | 281 | 57.2 | 891 | 219 | 7.6 | 1142 | ×1.28 | ×7.53 |
<!-- /table:vs-llamacpp-q27 -->

llama.cpp has a higher capacity on three cards: 7 slots of 126,976 tokens, 888,832 in total
(`docs/notes/measurements/2026-09-23-llamacpp-vs-vllm-3cards.md`). It processes the prompts of its slots one after another, so the
total PP does not grow with streams, and the streams that are ready nearly stop while another prompt is processed.

Other llama.cpp records of the same model, with other conditions, for orientation only (`docs/notes/machine-config.md`):

| Conditions | PP | TG |
|---|---:|---:|
| `llama-bench` on three cards, short context, pp2048 and tg64, 23 Sep 2026 | 429.5 | 20.5 |
| Two cards, real prompt of 9,320 tokens, context 131,072, q8_0 KV cache, no MTP, 17 Aug 2026 | 409.4 | 16.6 |
| The same with MTP | 270.8 | 27.3 |

## Against pipeline parallelism on three cards

The same fork on three cards, the third on a chipset PCIe 2.0 x1 slot. Pipeline parallelism moves stages through host memory
(`VLLM_PP_HOST_TRANSFER=1`).

<!-- table:vs-pp3 -->
| Model | Prompt × streams | TP=2, 2 cards: PP | TP=2, 2 cards: TG | Pipeline, 3 cards: PP | Pipeline, 3 cards: TG | PP ratio | TG ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B | 300 × 1 | 630 | 38.0 | 381 * | 20.5 * | ×1.66 | ×1.85 |
| Qwen3.8-27B | 32k × 1 | 555 | 36.7 | 685 | 19.2 | ×0.81 | ×1.91 |
| Qwen3.8-27B | 125k × 1 | 283 | 34.6 | 355 | 15.2 | ×0.80 | ×2.28 |
| Qwen3.8-27B | 125k × 2 | 281 | 57.2 | 413 * | 21.1 * | ×0.68 | ×2.71 |
| Gemma 4 26B-A4B, 3–4 Oct | 250k × 1 | 692 | 42.8 | 338 | 9.5 | ×2.04 | ×4.48 |
| Gemma 4 26B-A4B, 3–4 Oct | 125k × 4 | 555 | 135.5 | 1433 | 49.7 | ×0.39 | ×2.73 |
| Gemma 4 26B-A4B, 3–4 Oct | 64k × 8 | 858 | 198.7 | 2150 | 82.5 | ×0.40 | ×2.41 |
<!-- /table:vs-pp3 -->

- TP=2 decodes ×1.8–4.5 faster with one stream, and ×2.4–2.7 faster with several streams.
- Pipeline parallelism has the higher PP on long prompts of Qwen3.8-27B (ratios of ×0.68–0.81) and with many streams on Gemma 4 (×0.39
  and ×0.40). These two Gemma TP rows predate the HIP prefill kernel, while the pipeline rows use a prefill change (`VLLM_ROCM_MOE_CHUNK`)
  from a branch that is not part of this release. With the HIP prefill kernel, TP=2 is ×2 faster on one 250k prompt of Gemma 4.
- Pipeline parallelism holds more KV cache. Gemma 4: 1,232,968 tokens against 584,044 in the records of those days, and 933,930 in a start on the release tree.
- MTP cannot run with pipeline parallelism, see [KNOWN_ISSUES.md](KNOWN_ISSUES.md).
- On Qwen3.8-27B, 32k × 1 and 125k × 1 are from the release tree. Earlier records of the point 125k × 1 gave a TG of 10.3–14.1, so this number moves more than others. The row 125k × 2
  is from 23 Sep 2026. For 300 × 1, the release tree gave a TG of 21.4, but it was the first request after the start (TTFT 7.8 s), so the row keeps the earlier warm record.

## Scaling from one card to two

Qwen3.5-9B, window 131,072, 19 Sep 2026. Decode grows by ×1.7–1.8, and the KV cache from 326 to 1,240 blocks (×3.8) because each card
holds half of the weights.

<!-- table:tp-scaling -->
| Prompt × streams | 1 card PP | 1 card TG | 2 cards PP | 2 cards TG | PP ratio | TG ratio |
|---|---:|---:|---:|---:|---:|---:|
| 125k × 1 | 379 | 29.6 | 694 | 53.1 | ×1.83 | ×1.80 |
| 125k × 4 | 425 | 43.5 | 815 | 73.3 | ×1.92 | ×1.69 |
<!-- /table:tp-scaling -->

## KV cache format

Qwen3.6-35B-A3B, MTP 2, window 262,144. The int4 format holds ×1.8 more tokens, and it is much faster on long contexts because only int4
has HIP attention kernels. int8 goes through the general Triton kernel. This is a missing kernel and not a property of the format.

<!-- table:kv -->
| Qwen3.6-35B-A3B, MTP 2, context 262,144 | int4 KV | int8 KV | int4 / int8 |
|---|---:|---:|---:|
| KV cache, tokens | 634,664 | 353,324 | ×1.80 |
| TG 300 × 1 | 102.9 | 105.4 | ×0.98 |
| TG 32k × 1 | 126.0 | 105.8 | ×1.19 |
| TG 32k × 2 | 174.7 | 123.6 | ×1.41 |
| TG 125k × 1 | 118.6 | 52.2 | ×2.27 |
| TG 125k × 2 | 132.3 | 43.6 | ×3.03 |
| PP 32k × 1 | 1944 | 1401 | ×1.39 |
| PP 125k × 1 | 1210 | 547 | ×2.21 |
<!-- /table:kv -->

## Effect of each kernel

Each row is an A/B on one tree. Absolute values are not comparable between rows.

<!-- table:components -->
| Component | Metric | Prompt × streams | Without | With | Ratio |
|---|---|---|---:|---:|---:|
| HIP decode attention kernel, Qwen3.8-27B | TG | 125k × 1 | 25.1 | 31.0 | ×1.23 |
| HIP decode attention kernel, Qwen3.8-27B | TG | 125k × 2 | 36.6 | 49.7 | ×1.36 |
| HIP decode attention kernel, Qwen3.8-27B | TG | 32k × 4 | 77.5 | 90.6 | ×1.17 |
| HIP prefill kernel, Gemma 4 26B-A4B | PP | 32k × 1 | 981 | 1587 | ×1.62 |
| HIP prefill kernel, Gemma 4 26B-A4B | PP | 125k × 1 | 555 | 1068 | ×1.92 |
| HIP prefill kernel, Gemma 4 26B-A4B | PP | 250k × 1 | 317 | 692 | ×2.18 |
| HIP prefill kernel, Qwen3.6-35B-A3B | PP | 32k × 1 | 1638 | 2168 | ×1.32 |
| W8A16 kernel, Qwen3.6-35B-A3B | TG | 32k × 1 | 52.8 | 64.5 | ×1.22 |
| W8A16 kernel, Qwen3.6-35B-A3B | TG | 32k × 4 | 161.6 | 198.0 | ×1.23 |
| W8A16 kernel, Qwen3.6-35B-A3B | TG | 300 × 4 | 171.3 | 206.6 | ×1.21 |
| W8A16 kernel, repeat | TG | 32k × 1 | 52.3 | 64.1 | ×1.22 |
| W8A16 kernel, repeat | TG | 32k × 4 | 167.9 | 198.7 | ×1.18 |
| W8A16 kernel, repeat | TG | 300 × 4 | 173.5 | 205.7 | ×1.19 |
<!-- /table:components -->

Older kernel results (the int4 GEMM ladder from 9.6 to 52.1 TOPS, the matrix-vector kernel and tile sizes) are described in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Quality

Qwen3.6-35B-A3B. Kullback-Leibler divergence to AesSedai Q6_K in llama.cpp as the reference, 54,696 tokens (Ukrainian, English, code),
top-20 candidates and a merged tail, by `kld-cross-engine.py` of the working repository. Source:
`docs/notes/measurements/2026-10-07-qwen36-config-results.md`, 7 Oct 2026.

| Candidate | KLD, all | KLD, Ukrainian | KLD, English | KLD, code | Top-1 agreement |
|---|---:|---:|---:|---:|---:|
| GGUF Q5_K_M | 0.0090 | 0.0121 | 0.0064 | 0.0050 | 96.1 % |
| This fork, CT-INT4 + W8A16, int8 KV | 0.0216 | 0.0282 | 0.0165 | 0.0125 | 94.2 % |
| This fork, CT-INT4 + W8A16, int4 KV | 0.0271 | 0.0349 | 0.0210 | 0.0163 | 93.3 % |

The rows of the fork are upper estimates, because they include the difference between engines. This fork loses about three times more than
Q5_K_M. The int4 KV cache accounts for about a quarter of the loss; most of the rest comes from the 4-bit experts with group 32.

On Qwen3.5-9B, the int4 KV cache raised perplexity by 0.4 % (5.940 to 5.965). W4A4 in all layers raised it by 19.9 % and is off by default.

**Long context.** A number hidden in the prompt was retrieved in 16 of 16 runs at 119,444 tokens and in 7 of 7 runs at 199,158 tokens. The
eighth request of the second series was stopped by the test's time limit and not by the model.

## Repeats

Pairs of runs of the same or neighbouring code, for judging the error of the tables above. Note the spread of PP at 300 × 4: this is why
that point is not shown in the matrices.

<!-- table:repeats -->
| Pair of runs | Prompt × streams | TG, run A / run B | TG spread | PP, run A / run B | PP spread |
|---|---|---:|---:|---:|---:|
| Gemma 4 26B-A4B, same tree | 23k × 1 | 56.5 / 56.5 | 0.0 % | 1748 / 1733 | 0.9 % |
| Gemma 4 26B-A4B, same tree | 100k × 1 | 51.2 / 51.0 | 0.4 % | 1211 / 1212 | 0.0 % |
| Qwen3.6-35B-A3B, W8A16 on, same tree | 32k × 1 | 64.5 / 64.1 | 0.7 % | 2101 / 2139 | 1.8 % |
| Qwen3.6-35B-A3B, W8A16 on, same tree | 32k × 4 | 198.0 / 198.7 | 0.4 % | 2132 / 2133 | 0.1 % |
| Qwen3.6-35B-A3B, W8A16 on, same tree | 300 × 4 | 206.6 / 205.7 | 0.4 % | 1476 / 1722 | 15.4 % |
| Qwen3.6-35B-A3B, W8A16 off, same tree | 32k × 1 | 52.8 / 52.3 | 1.0 % | 2181 / 2186 | 0.2 % |
| Qwen3.6-35B-A3B, W8A16 off, same tree | 32k × 4 | 161.6 / 167.9 | 3.8 % | 2121 / 2141 | 0.9 % |
| Qwen3.6-35B-A3B, W8A16 off, same tree | 300 × 4 | 171.3 / 173.5 | 1.3 % | 1313 / 1818 | 32.3 % |
| Qwen3.8-27B, neighbouring commits | 300 × 1 | 38.1 / 37.2 | 2.4 % | 611 / 609 | 0.4 % |
| Qwen3.8-27B, neighbouring commits | 300 × 4 | 117.8 / 116.5 | 1.1 % | 704 / 738 | 4.7 % |
| Qwen3.8-27B, same tree | 300 × 1 | 38.1 / 37.2 | 2.4 % | 606 / 609 | 0.4 % |
| Qwen3.8-27B, same tree | 300 × 4 | 117.9 / 116.5 | 1.1 % | 336 / 738 | 74.8 % |
| Qwen3.8-27B, 1 Oct and release tree | 300 × 1 | 37.2 / 38.0 | 2.1 % | 609 / 630 | 3.5 % |
| Qwen3.8-27B, 1 Oct and release tree | 32k × 1 | 36.6 / 36.7 | 0.3 % | 556 / 555 | 0.0 % |
| Qwen3.8-27B, 1 Oct and release tree | 125k × 1 | 34.6 / 34.6 | 0.0 % | 284 / 283 | 0.2 % |
| Qwen3.8-27B, 1 Oct and release tree | 125k × 2 | 57.3 / 57.2 | 0.3 % | 295 / 281 | 5.1 % |
| Qwen3.6-35B-A3B, 7 Oct and release tree | 32k × 1 | 64.0 / 64.2 | 0.3 % | 2133 / 2094 | 1.8 % |
| Qwen3.6-35B-A3B, 7 Oct and release tree | 125k × 1 | 60.8 / 58.8 | 3.3 % | 1337 / 1289 | 3.7 % |
| Qwen3.6-35B-A3B, 7 Oct and release tree | 125k × 2 | 101.4 / 99.8 | 1.6 % | 1339 / 1348 | 0.7 % |
| Qwen3.8-27B, release tree, two starts | 300 × 1 | 37.6 / 38.0 | 1.0 % | 633 / 615 | 2.9 % |
| Qwen3.8-27B, release tree, two starts | 32k × 1 | 36.7 / 36.9 | 0.7 % | 556 / 556 | 0.0 % |
| Qwen3.6-35B-A3B + MTP, release tree, two starts | 300 × 1 | 104.1 / 107.2 | 3.0 % | 668 / 1003 | 40.1 % |
| Qwen3.6-35B-A3B + MTP, release tree, two starts | 32k × 1 | 123.7 / 96.5 | 24.7 % | 1933 / 1969 | 1.9 % |
| Qwen3.6-35B-A3B + MTP, release tree, two starts | 125k × 1 | 102.3 / 118.2 | 14.4 % | 1158 / 1211 | 4.5 % |
| Qwen3.6-35B-A3B + MTP, release tree, two starts | 125k × 2 | 149.0 / 150.9 | 1.2 % | 1241 / 1251 | 0.8 % |
<!-- /table:repeats -->

## Configuration of each run

<!-- table:configs -->
| Run | Model | Checkpoint | Weights | Engine | Cards | Context | KV | UTIL | Max streams | Code | Date |
|---|---|---|---|---|---:|---:|---|---:|---:|---|---|
| `rel-q27` | Qwen3.8-27B | philbert440/Qwen3.8-27B-W4A16-AWQ | W4A16 AWQ g128, 14.5 GiB | fork | 2 | 262,144 | int4 | 0.92 | 8 | 0.1.0 (vLLM tree ac6a7c2b83) | 2026-11-05 |
| `rel-q27-mtp` | Qwen3.8-27B | philbert440/Qwen3.8-27B-W4A16-AWQ | W4A16 AWQ g128, 14.5 GiB | fork | 2 | 262,144 | int4 | 0.92 | 8 | 0.1.0 (vLLM tree ac6a7c2b83) | 2026-11-05 |
| `rel-q9` | Qwen3.5-9B | atalsharma/Qwen3.5-9B-AWQ | W4A16 AWQ g128, 7.14 GiB | fork | 2 | 262,144 | int4 | 0.92 | 8 | 0.1.0 (vLLM tree ac6a7c2b83) | 2026-11-05 |
| `rel-q9-mtp` | Qwen3.5-9B | atalsharma/Qwen3.5-9B-AWQ + MTP head from Qwen/Qwen3.5-9B | W4A16 AWQ g128, 7.14 GiB | fork | 2 | 262,144 | int4 | 0.92 | 8 | 0.1.0 (vLLM tree ac6a7c2b83) | 2026-11-05 |
| `rel-q35` | Qwen3.5-35B-A3B | cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | MoE W4A16 AWQ | fork | 2 | 131,072 | int4 | 0.97 | 4 | 0.1.0 (vLLM tree ac6a7c2b83) | 2026-11-05 |
| `rel-q35-mtp` | Qwen3.5-35B-A3B | cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | MoE W4A16 AWQ | fork | 2 | 131,072 | int4 | 0.97 | 4 | 0.1.0 (vLLM tree ac6a7c2b83) | 2026-11-05 |
| `rel-q36` | Qwen3.6-35B-A3B | cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit, converted by scripts/awq-to-ct.py | MoE CT-INT4 + W8A16, 24 GB | fork | 2 | 131,072 | int4 | 0.97 | 8 | 0.1.0 (vLLM tree ac6a7c2b83) | 2026-11-05 |
| `q36-mtp` | Qwen3.6-35B-A3B | cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit, converted by scripts/awq-to-ct.py | MoE CT-INT4 + W8A16, 24 GB | fork | 2 | 131,072 | int4 | 0.97 | 4 | 2439264a0b | 2026-10-06 |
| `rel-q36-mtp-262k` | Qwen3.6-35B-A3B | cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit, converted by scripts/awq-to-ct.py | MoE CT-INT4 + W8A16, 24 GB | fork | 2 | 262,144 | int4 | 0.97 | 8 | 0.1.0 (vLLM tree ac6a7c2b83) | 2026-11-05 |
| `rel-gemma` | Gemma 4 26B-A4B | cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4 | MoE AWQ INT4 g32, 17.2 GB | fork | 2 | 262,144 | int4 | 0.97 | 16 | 0.1.0 (vLLM tree ac6a7c2b83) | 2026-11-05 |
| `rel-llama-q36` | Qwen3.6-35B-A3B | AesSedai Q5_K_M GGUF | GGUF Q5_K_M | llama.cpp | 2 | 262,144 | q8 | — | — | 058df67 | 2026-11-05 |
| `rel-llama-q27-3cards` | Qwen3.8-27B | unsloth Qwen3.8-27B-Q4_0.gguf | GGUF Q4_0, 14.94 GiB | llama.cpp | 3 | 126,976 | q4_0 | — | — | 058df67 | 2026-11-05 |
| `rel-pp3-q27` | Qwen3.8-27B | philbert440/Qwen3.8-27B-W4A16-AWQ | W4A16 AWQ g128, 14.5 GiB | fork, pipeline parallel 3 | 3 | 262,144 | int4 | 0.97 | 8 | 0.1.0 (vLLM tree ac6a7c2b83) | 2026-11-05 |
| `pp3-gemma` | Gemma 4 26B-A4B | cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4 | MoE AWQ INT4 g32, 17.2 GB | fork, pipeline parallel 3 | 3 | 262,144 | int4 | 0.95-0.97 | 16 | a8de020c41, eb6b697fbd | 2026-10-03, 2026-10-04 |
| `q27` | Qwen3.8-27B | philbert440/Qwen3.8-27B-W4A16-AWQ | W4A16 AWQ g128, 14.5 GiB | fork | 2 | 262,144 | int4 | 0.92 | 8 | 9243d6330b | 2026-10-01 |
| `q27-mtp` | Qwen3.8-27B | philbert440/Qwen3.8-27B-W4A16-AWQ | W4A16 AWQ g128, 14.5 GiB | fork | 2 | 262,144 | int4 | 0.92 | 8 | 9243d6330b | 2026-10-01 |
| `q9` | Qwen3.5-9B | atalsharma/Qwen3.5-9B-AWQ | W4A16 AWQ g128, 7.14 GiB | fork | 2 | 262,144 | int4 | 0.92 | 8 | e7e3cfe81a | 2026-09-29 |
| `q9-mtp` | Qwen3.5-9B | atalsharma/Qwen3.5-9B-AWQ + MTP head from Qwen/Qwen3.5-9B | W4A16 AWQ g128, 7.14 GiB | fork | 2 | 262,144 | int4 | 0.92 | 8 | e7e3cfe81a | 2026-09-29 |
| `q35` | Qwen3.5-35B-A3B | cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | MoE W4A16 AWQ | fork | 2 | 131,072 | int4 | 0.97 | 4 | 31f4535161 | 2026-10-01 |
| `q35-mtp` | Qwen3.5-35B-A3B | cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | MoE W4A16 AWQ | fork | 2 | 131,072 | int4 | 0.97 | 4 | 31f4535161 | 2026-10-01 |
| `q35-offload` | Qwen3.5-35B-A3B | cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit | MoE W4A16 AWQ | fork | 2 | 131,072 | int4 | 0.97 | 4 | 31f4535161 | 2026-10-01 |
| `q36` | Qwen3.6-35B-A3B | cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit, converted by scripts/awq-to-ct.py | MoE CT-INT4 + W8A16, 24 GB | fork | 2 | 131,072 | int4 | 0.97 | 4 | 2439264a0b | 2026-10-07 |
| `q36-mtp-262k` | Qwen3.6-35B-A3B | cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit, converted by scripts/awq-to-ct.py | MoE CT-INT4 + W8A16, 24 GB | fork | 2 | 262,144 | int4 | 0.97 | 4 | 2439264a0b | 2026-10-07 |
| `q36-mtp-262k-kv8` | Qwen3.6-35B-A3B | cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit, converted by scripts/awq-to-ct.py | MoE CT-INT4 + W8A16, 24 GB | fork | 2 | 262,144 | int8 | 0.97 | 4 | 2439264a0b | 2026-10-07 |
| `gemma` | Gemma 4 26B-A4B | cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4 | MoE AWQ INT4 g32, 17.2 GB | fork | 2 | 262,144 | int4 | 0.97 | 16 | e6393b86f0 | 2026-10-03 |
| `gemma-streams` | Gemma 4 26B-A4B | cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4 | MoE AWQ INT4 g32, 17.2 GB | fork | 2 | 262,144 | int4 | 0.97 | 16 | a8de020c41 | 2026-10-03 |
| `llama-q36` | Qwen3.6-35B-A3B | AesSedai Q5_K_M GGUF | GGUF Q5_K_M | llama.cpp | 2 | 262,144 | q8 | — | — | 058df67 | 2026-10-07 |
| `llama-q27-3cards` | Qwen3.8-27B | unsloth Qwen3.8-27B-Q4_0.gguf | GGUF Q4_0, 14.94 GiB | llama.cpp | 3 | 126,976 | q4_0 | — | — | 058df67 | 2026-09-23 |
| `pp3-q27` | Qwen3.8-27B | philbert440/Qwen3.8-27B-W4A16-AWQ | W4A16 AWQ g128, 14.5 GiB | fork, pipeline parallel 3 | 3 | 262,144 | int4 | 0.93-0.97 | 8 | 2026-09-23 and 2026-09-29 trees | 2026-09-23, 2026-09-29 |
<!-- /table:configs -->

The `Code` column is `0.1.0 (vLLM tree ac6a7c2b83)` for the release tree, otherwise a commit of the working fork, which cannot be fetched from upstream. All of these commits are ancestors of the release
tree, except `eb6b697fbd` (`VLLM_ROCM_MOE_CHUNK`, marked above). `2439264a0b` is the direct parent of the release commit.

`Context` is `--max-model-len` of the server, and `Max streams` is `--max-num-seqs`. `UTIL` is `--gpu-memory-utilization`.

### Common command

```bash
HIP_VISIBLE_DEVICES=0,1 ROCM_PATH=/opt/rocm PYTORCH_ROCM_ARCH=gfx1030 SAFETENSORS_FAST_GPU=1 OMP_NUM_THREADS=4 <settings below> \
vllm serve <model> --disable-custom-all-reduce --dtype float16 \
  --max-model-len <context> --kv-cache-dtype int4_per_token_head --gpu-memory-utilization <UTIL> \
  --max-num-seqs <streams> --max-num-batched-tokens 8192 --long-prefill-token-threshold 2048 \
  --attention-backend TRITON_ATTN --pipeline-parallel-size 1 --tensor-parallel-size 2
```

MTP adds `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":2}'` and an explicit list of
`cudagraph_capture_sizes`. `./serve.sh PRESET --print-command` prints the exact command of a preset.

<!-- table:flags -->
| Run | Settings beyond the common command |
|---|---|
| `rel-q27` | VLLM_ROCM_W4A8=256 VLLM_ROCM_W4A8_TILE=8 VLLM_ROCM_LMHEAD_INT8=1 VLLM_ROCM_EMBED_UVA=1 VLLM_ROCM_INT4_DECODE_QLEN_MAX=1 |
| `rel-q27-mtp` | as q27, VLLM_ROCM_INT4_DECODE_QLEN_MAX=3, MTP 2 drafts, capture sizes up to 48 |
| `rel-q9` | as q27 |
| `rel-q9-mtp` | as q27-mtp |
| `rel-q35` | VLLM_ROCM_MOE_RDNA2=1 |
| `rel-q35-mtp` | as q35, MTP 2 drafts, capture sizes up to 48 |
| `rel-q36` | VLLM_ROCM_MOE_RDNA2=1 VLLM_ROCM_W8A16=1 |
| `q36-mtp` | as q36, VLLM_ROCM_INT4_DECODE_QLEN_MAX=3, MTP 2 drafts, capture sizes up to 24 |
| `rel-q36-mtp-262k` | as q36-mtp |
| `rel-gemma` | VLLM_ROCM_MOE_RDNA2=1 VLLM_ROCM_INT4_PREFILL_KERNEL=hip |
| `rel-llama-q36` | q8 KV cache |
| `rel-llama-q27-3cards` | -sm layer, q4_0 KV (-fa on), -b 2048 -ub 512, pipeline mode, -np N, slot 126976 |
| `rel-pp3-q27` | pipeline parallelism on 3 cards, host transfer between stages |
| `pp3-gemma` | pipeline parallelism on 3 cards; cap-pp3b uses VLLM_ROCM_MOE_CHUNK from a branch that is not in this release |
| `q27` | VLLM_ROCM_W4A8=256 VLLM_ROCM_W4A8_TILE=8 VLLM_ROCM_LMHEAD_INT8=1 VLLM_ROCM_EMBED_UVA=1 VLLM_ROCM_INT4_DECODE_QLEN_MAX=1 |
| `q27-mtp` | as q27, VLLM_ROCM_INT4_DECODE_QLEN_MAX=3, MTP 2 drafts, capture sizes up to 48 |
| `q9` | as q27 |
| `q9-mtp` | as q27-mtp |
| `q35` | VLLM_ROCM_MOE_RDNA2=1; measured before the HIP prefill kernel was added |
| `q35-mtp` | as q35, MTP 2 drafts, capture sizes up to 48 |
| `q35-offload` | as q35, KV offload to RAM 32 GiB |
| `q36` | VLLM_ROCM_MOE_RDNA2=1 VLLM_ROCM_W8A16=1 |
| `q36-mtp-262k` | as q36-mtp |
| `q36-mtp-262k-kv8` | as q36-mtp, KV int8_per_token_head |
| `gemma` | VLLM_ROCM_MOE_RDNA2=1 VLLM_ROCM_INT4_PREFILL_KERNEL=hip |
| `gemma-streams` | VLLM_ROCM_MOE_RDNA2=1; before the HIP prefill kernel |
| `llama-q36` | q8 KV cache |
| `llama-q27-3cards` | -sm layer, q4_0 KV (-fa on), -b 2048 -ub 512, pipeline mode, -np N, slot 126976 |
| `pp3-q27` | pipeline parallelism on 3 cards, host transfer between stages |
<!-- /table:flags -->

## Other records of this project

Earlier records of Qwen3.8-27B from 28 Sep 2026 (commit `e7e3cfe81a`) are in `docs/notes/measurements/2026-09-28-mtp-attention-results.md`. This document uses the
records of the release tree. The later commits run the HIP attention kernel under MTP (`VLLM_ROCM_INT4_DECODE_QLEN_MAX=3`) and add kernels
for small batches, so TG at 125k × 1 is 31.0 in the earlier records and 34.6 here, and with MTP 34.4 and 63.5. TG with MTP at 300 × 1 is
68.6 in those records, 54.7 on 1 Oct 2026 and 73.0 here, which falls under the caution about MTP on random tokens.

The comparison with llama.cpp on three cards in the talk of 5 Nov 2026 uses the same-day records of 23 Sep 2026, where pipeline parallelism gives a TG of 10.5 at
125k × 1. The table above has the record of the release tree, 15.2.

## Reproducing

```bash
./serve.sh qwen3.8-27b --no-mtp
bench/run-matrix.sh qwen3.8-27b --url http://127.0.0.1:8100 --label my-machine
./stop.sh
```

To repeat one row exactly, pass the `Context`, `Max streams` and `UTIL` of its row in [the configuration table](#configuration-of-each-run), for example `./serve.sh qwen3.6-35b-a3b --no-mtp --ctx 131072 --streams 4`. Numbers from another machine or day are comparable only with the same state of the instrument.

The record format and the table generator are described in [results/README.md](../results/README.md). `python3 results/make_tables.py --check` verifies
that every table in this file matches the records.

### The llama.cpp rows

llama.cpp `058df67`, all layers on the GPUs. These commands were run on 5 Nov 2026 for the tables above.

```bash
# Qwen3.6-35B-A3B, Q5_K_M from AesSedai, two cards
HIP_VISIBLE_DEVICES=0,1 llama-server -m <Q5_K_M.gguf> --alias q36 --host 127.0.0.1 --port 8204 -c 262144 -np 4 --kv-unified \
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 --jinja --metrics --reasoning-format deepseek

# Qwen3.8-27B, Q4_0 from unsloth, three cards. N is the number of streams, with one start per point
HIP_VISIBLE_DEVICES=0,1,2 llama-server --model <Q4_0.gguf> --n-gpu-layers 99 --split-mode layer --flash-attn on \
  --cache-type-k q4_0 --cache-type-v q4_0 --parallel N --ctx-size $((N * 126976)) --fit off \
  --metrics --host 127.0.0.1 --port 8101

# The same probe as for the fork
timeout 1500 python3 bench/kv-concurrency.py --engine llamacpp --base-url http://127.0.0.1:8204 --streams 2 --length 125000 \
  --blocks 0 --max-tokens 256 --request-timeout 1400 --output my-llama.jsonl --label my-llama
```

### The pipeline rows

`serve.sh` always sets TP to the number of cards, so pipeline parallelism needs a manual command. This one was run on the release tree on 5 Nov 2026
for the points 300 × 1, 32k × 1 and 125k × 1. The start took 19 minutes.

```bash
PATH=<venv>/bin:/opt/rocm/bin:$PATH HIP_VISIBLE_DEVICES=0,1,2 ROCM_PATH=/opt/rocm PYTORCH_ROCM_ARCH=gfx1030 SAFETENSORS_FAST_GPU=1 VLLM_PP_HOST_TRANSFER=1 \
VLLM_ROCM_W4A8=256 VLLM_ROCM_W4A8_TILE=8 VLLM_ROCM_LMHEAD_INT8=1 VLLM_ROCM_EMBED_UVA=1 VLLM_ROCM_MIN_P_FLOOR=0.05 \
<venv>/bin/vllm serve <Qwen3.8-27B-W4A16-AWQ> --disable-custom-all-reduce --dtype float16 \
  --max-model-len 262144 --kv-cache-dtype int4_per_token_head --gpu-memory-utilization 0.97 \
  --max-num-seqs 8 --max-num-batched-tokens 8192 --long-prefill-token-threshold 2048 \
  --attention-backend TRITON_ATTN --no-async-scheduling --pipeline-parallel-size 3 --tensor-parallel-size 1
```

Four streams used `--gpu-memory-utilization 0.93`: at 0.97 the third stage ran out of memory. The records of 29 Sep 2026 also set
`VLLM_PP_LAYER_PARTITION=24,24,16`.
