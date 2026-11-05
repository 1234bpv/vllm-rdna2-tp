# vLLM on RDNA2: 60+ tokens/s on long contexts with two consumer AMD cards

A fork of [vLLM](https://github.com/vllm-project/vllm) that serves current quantized LLMs on two consumer AMD RDNA2 cards
(RX 6800, RX 6800 XT; chip gfx1030) with tensor parallelism. **Qwen3.8-27B with MTP decodes at 60+ tokens/s with a prompt of
125,000 tokens in a context window of 262,144 tokens, with the GPU clock limited to an economical 1600 MHz.** The speed comes
from HIP kernels for the quantized matrix products and attention, and from an all-reduce through host memory, so the two cards
do not need P2P access to each other. Nothing was measured at stock clocks.

| What was measured | Result |
|---|---|
| Qwen3.8-27B + MTP, window 262,144 on 2 × 16 GB, prompts of 32,768 and 125,000 random tokens | decode **66.0** and **63.5 tokens/s**, prompt processing 533 and 271 tokens/s |
| The same without MTP, prompt of 125,000 tokens | 34.6 tokens/s, prompt processing 283 tokens/s |
| Qwen3.6-35B-A3B, prompt of 125,000 tokens, window 131,072 | decode **58.8 tokens/s**, prompt processing **1,289 tokens/s** |
| The same point in llama.cpp on the same two cards, Q4_K_M | 20.4 tokens/s, 839 tokens/s |
| Qwen3.8-27B in llama.cpp on the same two cards, Q4_0, prompt of 125,000 tokens | 7.1 tokens/s, 317 tokens/s: higher prompt processing than this fork, a fifth of its decode |
| Decode with this all-reduce against RCCL, direct A/B | about **+7 %** (6.6–7.9 %) with one stream on Qwen3.8-27B and Qwen3.5-9B; no measurable gain with several streams or on Qwen3.6-35B-A3B |
| One exchange between the cards, decode size (5,120 fp16 elements), microbenchmark in a loop | **17 µs**, against about 164 µs through RCCL. This ×9.6 does not carry over to a decode step |

The project is not affiliated with AMD or with the vLLM project. AMD does not list the RX 6800 or RX 6800 XT as supported
by ROCm, and the vLLM documentation does not name this target. Use it at your own risk: a random GPU fault stopped 4 of 27
starts on the reference machine, and each time a reboot followed, see [Main limitations](#main-limitations).

## Contents

- [Why a host-memory all-reduce](#why-a-host-memory-all-reduce)
- [Performance](#performance)
- [Verification status of this release](#verification-status-of-this-release)
- [Tested hardware and models](#tested-hardware-and-models)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Using the server](#using-the-server)
- [Quantizations and context lengths](#quantizations-and-context-lengths)
- [Main limitations](#main-limitations)
- [Repository layout](#repository-layout)
- [Reproducing the numbers](#reproducing-the-numbers)
- [Similar projects](#similar-projects)
- [Citing](#citing)
- [License and attribution](#license-and-attribution)

## Why a host-memory all-reduce

Tensor parallelism (TP) splits every weight matrix across the cards, so the cards must exchange results many times per
generated token: about 130 all-reduces per token on Qwen3.8-27B. On these consumer cards there is no P2P path
(`torch.cuda.can_device_access_peer(0, 1)` is false), so RCCL copies through host memory with CPU helper threads, and even
the smallest exchange costs more than 100 µs in a microbenchmark. The fast all-reduce paths of upstream vLLM assume direct access to the other
card's memory, or data-center targets.

This fork adds a small HIP kernel, `csrc/rocm/host_all_reduce.cu`. Each rank writes its vector into its own slot of shared
host memory, publishes a step number, waits for the other rank and adds. The step number is kept in memory and not in a
kernel argument, so the kernel also works inside captured CUDA graphs.

<!-- table:all-reduce -->
| fp16 elements | Size | This fork, µs | RCCL, µs | RCCL / fork |
|---|---:|---:|---:|---:|
| 5,120 (decode exchange) | 10 KiB | 17.1 | ~164 | ×9.6 |
| 10,240 | 20 KiB | 22.6 | ~128 | ×5.6 |
| 40,960 | 80 KiB | 64.5 | ~128 | ×2.0 |
<!-- /table:all-reduce -->

Measured in a loop on the release tree on 5 Nov 2026, medians of 6 runs. Read this ratio as a microbenchmark only: with about 130
exchanges per token it would predict saving about 19 ms of a 28 ms step on Qwen3.8-27B, and the direct A/B below shows about 1.8 ms.
RCCL inside a decode step was not timed separately. Above about 262,144 elements RCCL was faster in an earlier
measurement, so large exchanges still go through it. The kernel handles exactly two ranks and fp16 data.

**What this gives end to end.** A faster exchange does not multiply the generation speed. The direct A/B below runs the same
matrix with `VLLM_ROCM_HOST_ALL_REDUCE=0` (RCCL) and `=1` (default):

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

With one stream, decode gains about 7 % (6.6–7.9 %) on Qwen3.8-27B and, in one run per mode, on Qwen3.5-9B. With several streams and on Qwen3.6-35B-A3B the differences
are between −3 % and +3 %, which is within the spread of repeats, so no gain is shown there. Prompt processing does not change.
The speed against llama.cpp below therefore comes mostly from the kernels described in
[ARCHITECTURE.md](docs/ARCHITECTURE.md), not from the exchange. See the full
[all-reduce measurements](docs/PERFORMANCE.md#all-reduce).

**Correctness.** On the tested pair, the RCCL `all_gather` and `reduce_scatter` return wrong data on one rank in about half
of process starts, and the outcome is fixed when the communicator is created. The fork therefore builds these collectives
from all-reduce (`VLLM_ROCM_COLLECTIVES_VIA_ALLREDUCE=1`, on by default). A dedicated test of the host all-reduce
(`bench/hostar_test.py`) found 0 wrong exchanges out of 1,800, and 0 of 150 in each of 11 runs on the release tree.

## Performance

Measured on one machine, described [below](#tested-hardware-and-models). All prompts are random tokens, output is 256
tokens, temperature 0.

- **PP**: prompt processing, tokens/s, the sum of the prompt tokens divided by the time until the last stream got its
  first token.
- **TG**: decode, tokens/s, in the window when all streams generate. **ΣTG** is the total of all streams.
- **TTFT**: time to the first token of the last stream.
- **MTP**: multi-token prediction, the model's speculative decoding, with 2 draft tokens.

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

Everything was measured on the release tree on 5 Nov 2026, unless a cell is marked `*`: such a point was not repeated and comes
from an earlier run. The Qwen3.6-35B-A3B + MTP row with the window 131,072 is from 6–7 Oct 2026, on the direct parent of the
release tree.

Read these numbers with four cautions:

1. **The gain of MTP falls with the number of streams.** On Qwen3.8-27B at 32k it is 1.8× with one stream, 1.6× with two and 1.1× with four.
   On a real agent workload with 4 clients, MTP reduced the wall time of Qwen3.6-35B-A3B by 5.6 % (189.8 to 179.1 s).
   One stream on a real workload was not measured.
2. **Repeats without MTP.** TG repeats within 0–3.8 % and PP on long prompts within 0–5.1 %. Do not interpret smaller
   differences. PP of short prompts depends on warm-up: it is not shown with several streams, where repeats differ by up to
   75 %, and the 211 of Gemma 4 is the first point after a start.
3. **Repeats with MTP spread much more.** Two starts of the release tree gave a TG of 123.7 and 96.5 at the same point
   (24.7 %), because the share of accepted drafts depends on the random prompt.
4. **The KV cache size changes between starts.** The same preset gave 584,782 to 712,493 tokens (Qwen3.5-35B-A3B), and 268,101
   and 205,544 with MTP. The cause is not established. Plan for the smaller value ([KI-39](docs/KNOWN_ISSUES.md)).

See [the configuration of each row](docs/PERFORMANCE.md#configuration-of-each-run).

### Against llama.cpp, same two cards

Qwen3.6-35B-A3B, TP=2, no MTP, 4-bit expert weights and int4 KV cache, against llama.cpp 058df67 with Q4_K_M and a q8 KV
cache, no MTP. Both use 4-bit weights, in different formats. The quality of this Q4_K_M was not measured. Q5_K_M runs within
5 % in PP and 10 % in TG of it, and the [details](docs/PERFORMANCE.md#against-llamacpp) include Qwen3.8-27B, where llama.cpp
has the higher PP at 125,000 tokens.

<!-- table:vs-llamacpp-q4 -->
| Prompt × streams | Fork PP | Fork TG | Fork TTFT, s | llama.cpp PP | llama.cpp TG | llama.cpp TTFT, s | PP ratio | TG ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 × 1 | — | 66.1 | 0.3 | — | 59.5 | 1.0 | — | ×1.11 |
| 300 × 4 | — | 206.3 | 0.8 | — | 144.7 | 1.2 | — | ×1.43 |
| 32k × 1 | 2094 | 64.2 | 16 | 1341 | 38.2 | 24 | ×1.56 | ×1.68 |
| 32k × 2 | 2167 | 113.7 | 30 | 987 | 60.8 | 66 | ×2.20 | ×1.87 |
| 125k × 1 | 1289 | 58.8 | 97 | 839 | 20.4 | 149 | ×1.54 | ×2.88 |
| 125k × 2 | 1348 | 99.8 | 185 | 456 | 28.4 | 548 | ×2.96 | ×3.52 |
<!-- /table:vs-llamacpp-q4 -->

### Against pipeline parallelism on three cards

Tensor parallelism on two cards decodes faster than pipeline parallelism on three cards of the same fork. Pipeline
parallelism has the higher PP on long prompts in most rows, and it holds more KV cache. The Gemma 4 rows are from 3–4 Oct 2026, and two
of them predate the HIP prefill kernel. The ratios are TP=2 to pipeline; the
notes on these rows are in [PERFORMANCE.md](docs/PERFORMANCE.md#against-pipeline-parallelism-on-three-cards).

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

The full matrices for each model, the effect of each kernel, KV cache formats, scaling from one card to two, and quality
are in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Verification status of this release

This table says what has been checked for version 0.1.0 and what has not.

| Check | Status |
|---|---|
| The patch applied to upstream commit `c296851a7d` gives exactly the release tree | Verified: the tree hash matches `ac6a7c2b83b9` |
| Install, start and stop scripts: logic, failure paths, safety rules | Verified by `tests/` with stubs, without GPUs |
| Result tables match the measurement records | Verified: `python3 results/make_tables.py --check` |
| Weights named below are public and not gated | Verified on 4 Nov 2026 through the Hugging Face API |
| `scripts/add-mtp-head.py` against the public repositories | Verified on 4 Nov 2026: 15 tensors, 464 MiB; a repeated run changes nothing. Serving the result on the release tree was not run |
| `install.sh` from zero with the cards | Verified on 5 Nov 2026, in a clean Ubuntu 24.04 container following `docs/INSTALL.md` literally, and on the reference machine. Only this machine |
| Each preset started on the release tree | Verified on 5 Nov 2026: all five presets, and MTP on the four that have it, with `scripts/smoke-test.py` passing. See [the start failures](#main-limitations) |
| Performance tables on the release tree | Verified on 5 Nov 2026, from a fresh virtual environment. Points that were not repeated are marked `*` |
| llama.cpp and pipeline commands in `docs/PERFORMANCE.md` | Run on 5 Nov 2026 as written |
| `scripts/kill-stuck-rank.sh --force` | Run on 5 Nov 2026 on a real GPU process with 1.2 GB of VRAM held with SIGSTOP: SIGKILL sent, driver check passed, no page faults. Not run on a rank stuck after a real fault. `serve.sh --kill-stuck`: tests with stubs only |
| `constraints/rocm-7.2.txt` | Verified on 5 Nov 2026: `pip freeze` of a fresh install matches all 211 pins. The older environment behind the earlier records has many other versions, and Qwen3.8-27B gave the same table on both |

The release tree differs from the parent used for the 6–7 Oct rows of Qwen3.6-35B-A3B only by removed diagnostic leftovers and
changed default paths for diagnostic output.

## Tested hardware and models

**Reference machine.** AMD Radeon RX 6800 (60 CU) and RX 6800 XT (72 CU), 16 GB each, no P2P; 2 × Intel Xeon E5-2699 v3;
128 GB DDR4-2133; GPU clock limited to 1600 MHz; Kubuntu 24.04, Linux 6.8 with the in-kernel amdgpu driver; ROCm 7.2.4;
Python 3.12; PyTorch 2.13.0+rocm7.2; Triton 3.7.1. No other machine has been tested. Other gfx1030 cards (RX 6900 XT, PRO
W6800, PRO V620) have not been tested. Other RDNA2 chips (gfx1031 and later) use different limits and have not been tested.

| Preset | Model | Weights on Hugging Face | Measured with | MTP |
|---|---|---|---|---|
| `qwen3.8-27b` | Qwen3.8-27B | [`philbert440/Qwen3.8-27B-W4A16-AWQ`](https://huggingface.co/philbert440/Qwen3.8-27B-W4A16-AWQ) | TP=2, window 262,144 | yes |
| `qwen3.5-9b` | Qwen3.5-9B | [`atalsharma/Qwen3.5-9B-AWQ`](https://huggingface.co/atalsharma/Qwen3.5-9B-AWQ) | TP=2, window 262,144; prompts up to 8,192 | yes, the head is added by `scripts/add-mtp-head.py` |
| `qwen3.5-35b-a3b` | Qwen3.5-35B-A3B | [`cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit`](https://huggingface.co/cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit) | TP=2, window 131,072 | yes |
| `qwen3.6-35b-a3b` | Qwen3.6-35B-A3B | [`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`](https://huggingface.co/cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit), converted locally | TP=2, windows 131,072 and 262,144 | yes, on by default |
| `gemma-4-26b-a4b` | Gemma 4 26B-A4B | [`cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4`](https://huggingface.co/cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4) | TP=2, window 262,144 | no |

Other checks of Qwen3.6-35B-A3B on the reference machine: retrieval of a number hidden in the prompt worked in 16 of 16 runs
at 119,444 tokens and 7 of 7 runs at 199,158 tokens; an agent task set with 4 clients finished 18 of 18 tasks.

## Requirements

| Item | Tested | Notes |
|---|---|---|
| GPUs | RX 6800 + RX 6800 XT | Two gfx1030 cards for TP=2. One card (TP=1) was measured only with Qwen3.5-9B |
| OS | Kubuntu 24.04 | Linux only. `scripts/doctor.sh` warns on anything else |
| Kernel | 6.8, in-kernel amdgpu | ROCm is installed without DKMS |
| ROCm | 7.2.4 | `hipcc` must stay installed: HIP kernels compile while the server runs |
| Python | 3.12 | The installer accepts 3.10–3.14 with a warning |
| Host memory | 128 GB | The installer warns below 32 GB. The real minimum has not been measured |
| `/dev/shm` | see notes | The doctor warns below 8 GiB free. KV offload needs more |
| Disk | see notes | The doctor warns below 40 GiB free. Downloads of weights take 8–25 GB per model, and the Qwen3.6 preset needs 50 GB because it keeps the download and the converted copy. PyTorch alone is a 6.2 GB download |
| Build | about 15 min | The whole `install.sh` on the reference machine, with the native part of vLLM. The venv takes 17 GB |
| Network | needed for install | GitHub, PyPI, `download.pytorch.org`, Hugging Face |

Install ROCm first. These commands were used on the reference machine:

```bash
curl -LO https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/noble/amdgpu-install_7.2.4.70204-1_all.deb
sudo apt-get install -y ./amdgpu-install_7.2.4.70204-1_all.deb
sudo apt-get update
sudo amdgpu-install --usecase=rocm --no-dkms -y
sudo usermod -aG render,video "$USER"    # then log out and back in
```

## Quick start

Prerequisites: ROCm and the system packages from [INSTALL.md](docs/INSTALL.md), and about 75 GB of free disk for the commands below: 17 GB of venv, 7 GB of pip cache, 49 GB of weights.

```bash
git clone https://github.com/1234bpv/vllm-rdna2-tp.git vllm-rdna2-tp
cd vllm-rdna2-tp

scripts/doctor.sh --pre                       # check the system; nothing is changed
./install.sh                                  # downloads PyTorch (6.2 GB), builds vLLM
scripts/download-model.sh qwen3.6-35b-a3b     # downloads 25 GB of weights, then converts them locally (24 GB more)
./serve.sh qwen3.6-35b-a3b                    # the first start takes several minutes
python3 scripts/smoke-test.py                 # end-to-end check of the live server
./stop.sh
```

`install.sh` installs into `~/.local/share/vllm-rdna2` (change it with `--prefix` or `VLLM_RDNA2_HOME`). It never uses
`sudo`. It clones upstream vLLM at commit `c296851a7d`, applies `patches/vllm-rdna2-tp-0.1.0.patch` and builds in place. The
installation has to stay editable: the `.hip` kernels are compiled from the source tree at run time.

`serve.sh` checks the driver, retries a failed start, removes orphaned shared-memory files and sends a first test request,
because a known intermittent fault can appear on the first request. See [INSTALL.md](docs/INSTALL.md) and
[CONFIGURATION.md](docs/CONFIGURATION.md).

> **Never stop the server with `kill -9`.** Killing a process with pending GPU address faults can hang
> `kfd_process_wq_release` in the driver and freeze the machine. Use `./stop.sh`. If processes survive, see
> [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## Using the server

The server speaks the OpenAI API, by default at `http://127.0.0.1:8100/v1`, **without authentication**. Do not bind it to a
public address.

```bash
curl -s http://127.0.0.1:8100/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.6-35b-a3b",
  "messages": [{"role": "user", "content": "Explain tensor parallelism in two sentences."}],
  "max_tokens": 256
}'
```

Common options:

```bash
./serve.sh qwen3.8-27b --mtp                  # enable MTP for one stream
./serve.sh qwen3.8-27b --no-mtp --streams 8   # several streams: MTP off
./serve.sh qwen3.6-35b-a3b --ctx 131072       # shorter window, more room for the KV cache
./serve.sh qwen3.8-27b --util 0.86            # a display is attached to a compute card
./serve.sh qwen3.8-27b --print-command        # show the exact command and exit
./serve.sh --list                             # list the presets
```

**MTP or not:** MTP helps one stream on a long prompt the most. With several streams the gain falls, and on some points MTP
is slower. For interactive agent work, measure your own workload.

**During prompt processing the server sends nothing.** Set client timeouts longer than the longest prefill: a prompt of
125,000 tokens takes 92–461 s depending on the model, and aborting a request during prefill while MTP is on can hang the
workers ([KI-9](docs/KNOWN_ISSUES.md)).

## Quantizations and context lengths

| Layer | Format | Kernel | Status |
|---|---|---|---|
| Linear layers | W4A16 AWQ, group 128 or 32 | HIP and Triton, tiles tuned for gfx1030 | Used by all presets |
| Linear layers, prefill | W4A8 (`VLLM_ROCM_W4A8=256`) | HIP, `v_dot4` | Used by the 27B and 9B presets |
| Linear layers | W4A4 | HIP, `v_dot8` | Fastest, off: it raised perplexity by 19.9 % on 9B |
| Unquantized layers | W8A16 (`VLLM_ROCM_W8A16=1`) | HIP | Used by the Qwen3.6 preset |
| `lm_head` | fp16 or int8 (`VLLM_ROCM_LMHEAD_INT8=1`) | matrix-vector kernel at 89 % of the memory roofline | |
| MoE experts | AWQ int4, compressed-tensors (`VLLM_ROCM_MOE_RDNA2=1`) | HIP | Used by the 35B and Gemma presets |
| KV cache | int4 per token and head | HIP decode and prefill attention | Default, needed above 131,072 tokens |
| KV cache | int8 per token and head | Triton | Holds fewer tokens, ×2.3–3 slower at 125,000 tokens |
| Weights in `bfloat16` | not supported | | RDNA2 has no bf16 instructions: always `--dtype float16` (the scripts set it) |
| GGUF | not in this release | | |

Tested windows are listed [above](#tested-hardware-and-models). Larger windows depend on the free VRAM: a display on a
compute card takes 0.35–0.69 GiB in the reference setup. The start message states the limit when the window does not fit.

## Main limitations

- **Starts can fail at random, and then a stuck rank remains.** `hipErrorIllegalAddress` occurs at start or on the first
  request. In the release check it happened in 4 of 27 starts. Each time the other rank stayed in a GPU wait and ignored SIGTERM, so
  `serve.sh` stopped with code 6 instead of retrying. The next start after a reboot worked. The root cause is open;
  `AMD_SERIALIZE_KERNEL=3` removed it in one test (0 of 20), cost not measured.
- **Removing a stuck rank without a reboot is opt-in.** `serve.sh --kill-stuck` and `scripts/kill-stuck-rank.sh --force PID` send
  SIGKILL to a rank that ignored SIGTERM, if the kernel log has no page fault from it or on its card. In this release they were run
  on a real GPU process held with SIGSTOP, not on a rank stuck after a real fault. See
  [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md#the-process-does-not-exit).
- **Never use `kill -9` directly.** It can freeze the machine (KI-2).
- **The fast all-reduce handles exactly two ranks.** With another rank count the exchange goes through RCCL.
- **Pipeline parallelism and MTP cannot be combined:** the output is corrupted, so the start is refused.
- **MTP can hang on a client abort during prefill.** Use long client timeouts or run without MTP.
- **Keep the GPU order ascending.** `HIP_VISIBLE_DEVICES=1,0` hangs the first decode step.
- **The server has no authentication.**
- **Code comments in the patch are in Ukrainian,** as are the working notes in `docs/notes/`.

The complete list is in [KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md), with workarounds and sources.

## Repository layout

```
install.sh, serve.sh, stop.sh      install and run
scripts/                           doctor, driver check, model download, MTP head, converter, systemd unit, smoke test
presets/                           settings of each tested model
constraints/                       pinned Python packages
patches/                           the patch against upstream vLLM and the list of changed files
src/                               the main new files, readable without applying the patch
bench/                             benchmarks: throughput matrix, all-reduce latency, retrieval
kernels/                           standalone HIP microbenchmarks of the dot instructions and the int4 GEMM
contrib/system/                    machine notes: systemd units and kernel settings of the reference machine
results/                           measurement records and the generator of all tables
docs/                              documentation, see docs/README.md
tests/                             tests of the scripts and the table generator, no GPUs needed
```

## Reproducing the numbers

```bash
./serve.sh qwen3.8-27b --no-mtp
bench/run-matrix.sh qwen3.8-27b --url http://127.0.0.1:8100 --label my-machine
```

The records are written to `results/local/`, which git ignores.

Every table in this file and in `docs/PERFORMANCE.md` is generated from `results/kv-concurrency.jsonl` by
`results/make_tables.py`. See [results/README.md](results/README.md) and [bench/README.md](bench/README.md). Results from
other machines are welcome: please include the card models, clocks, ROCm version and the commit.

## Similar projects

Other public projects work on gfx1030. Their numbers are self-reported and were not reproduced here.

- [opengfx1030/vllm-rdna](https://github.com/opengfx1030/vllm-rdna): a larger vLLM port for the PRO V620. Its fast all-reduce
  requires P2P. If your cards have P2P, look there first.
- [stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts): gfx1030 patches for llama.cpp.

The difference here is the combination: tensor parallelism in vLLM with an own all-reduce through host memory on cards
without P2P. More in [RELATED_WORK.md](docs/RELATED_WORK.md).

## Citing

See [CITATION.cff](CITATION.cff). The work was presented as: S. Pogorilyy, P. Biletskyi. *Kernel-Level Adaptation of vLLM for
AMD RDNA2 Consumer GPUs.* International conference "53 Years of Scientific Cooperation", HLRS, Stuttgart, 5 November 2026.

## License and attribution

Apache License 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). This work is based on
[vLLM](https://github.com/vllm-project/vllm), commit `c296851a7d173fa89d2eefbca0243be42ae9b5e0`, copyright the vLLM
contributors. The files changed against that commit are listed in `patches/MODIFIED_FILES.txt`.
