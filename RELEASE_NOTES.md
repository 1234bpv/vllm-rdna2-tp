# vLLM on RDNA2 0.1.0

A fork of vLLM that serves current quantized LLMs on two consumer AMD RDNA2 cards (RX 6800, RX 6800 XT; chip gfx1030) with tensor parallelism. **Qwen3.8-27B with MTP decodes at 60+ tokens/s with a prompt of 125,000 tokens in a context window of 262,144 tokens, with the GPU clock limited to 1600 MHz.** The two cards do not need P2P access to each other.

## Main results

Reference machine: RX 6800 + RX 6800 XT, 16 GB each, GPU clock limited to 1600 MHz, ROCm 7.2.4.

| What was measured | Result |
|---|---|
| Qwen3.8-27B + MTP, window 262,144, prompts of 32,768 and 125,000 random tokens | decode 66.0 and 63.5 tokens/s, prompt processing 533 and 271 tokens/s |
| One exchange between the cards, decode size, microbenchmark in a loop | 17 µs, against about 164 µs through RCCL |
| Decode with this all-reduce against RCCL, direct A/B | about +7 % (6.6–7.9 %) with one stream on Qwen3.8-27B and Qwen3.5-9B; no measurable gain with several streams or on Qwen3.6-35B-A3B |
| Qwen3.6-35B-A3B, prompt of 125,000 tokens, TP=2 | prompt processing 1,289 tokens/s, decode 58.8 tokens/s |
| The same point in llama.cpp on the same cards, Q4_K_M | 839 tokens/s, 20.4 tokens/s |
| Qwen3.8-27B in llama.cpp on the same two cards, Q4_0, prompt of 125,000 tokens | 317 tokens/s, 7.1 tokens/s |
| Qwen3.8-27B, window 262,144, prompt of 125,000 tokens | 283 tokens/s, 34.6 tokens/s |

A faster exchange does not multiply the speed, and most of the gain against llama.cpp comes from the kernels. The full tables, with the configuration of each row and the cautions on MTP, are in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Tested models

Qwen3.8-27B, Qwen3.5-9B, Qwen3.5-35B-A3B, Qwen3.6-35B-A3B and Gemma 4 26B-A4B, with a preset for each.

## Install

```bash
git clone https://github.com/1234bpv/vllm-rdna2-tp.git vllm-rdna2-tp && cd vllm-rdna2-tp
scripts/doctor.sh --pre
./install.sh
scripts/download-model.sh qwen3.6-35b-a3b
./serve.sh qwen3.6-35b-a3b
```

ROCm has to be installed first: [docs/INSTALL.md](docs/INSTALL.md).

## What was and was not verified for this version

See [the verification status](README.md#verification-status-of-this-release). In short: on 5 Nov 2026 `install.sh` ran from zero on the reference machine and in a clean Ubuntu 24.04 container, all five presets started and passed the smoke test, and the tables were measured again on the release tree, with the points that were not repeated marked `*`.

**Only one machine has been tested.** Expect that something may need adjusting on yours, and please report it.

## Known problems

- A random fault can occur at start or on the first request: 4 of 27 starts in the release check, each followed by a reboot. The root cause is open. `serve.sh --kill-stuck` can remove the stuck rank instead (opt-in, SIGKILL behind checks of the kernel log), and this path has not yet met a real fault.
- The KV cache size changes between starts of the same preset.
- With MTP on, a client abort during prefill can hang the workers.
- **Never use `kill -9` directly**: it can hang the driver. Use `./stop.sh`.
- The server has no authentication.

The complete list is in [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md).

## License

Apache-2.0. A derivative work of [vLLM](https://github.com/vllm-project/vllm), commit `c296851a7d`. See [NOTICE](NOTICE).
