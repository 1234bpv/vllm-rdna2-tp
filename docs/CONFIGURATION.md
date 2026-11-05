# Configuration

Settings come from three places. A later place overrides an earlier one:

1. The preset, a file in `presets/`.
2. The options of `serve.sh`.
3. Arguments after `--`, passed unchanged to `vllm serve`.

The exception is `--pipeline-parallel-size` and `--tensor-parallel-size`: `serve.sh` always puts them last, so `--` cannot change them.
TP always equals the number of cards in `--gpus`.

To see the result without starting anything:

```bash
./serve.sh qwen3.8-27b --ctx 131072 --print-command
```

## Options of `serve.sh`

```
./serve.sh PRESET [options] [-- extra vllm serve arguments]
./serve.sh --list
./serve.sh --help
```

| Option | Effect | Default |
|---|---|---|
| `--model PATH\|HF_ID` | Weights: a local directory or a Hugging Face id | The local directory of the preset, else its `HF_ID` |
| `--mtp`, `--no-mtp` | Speculative decoding with the MTP head of the model, 2 draft tokens | From the preset |
| `--ctx N` | Context window in tokens (`--max-model-len`) | From the preset |
| `--streams N` | Concurrent sequences (`--max-num-seqs`) | From the preset |
| `--kv int4\|int8\|auto` | KV cache format | From the preset |
| `--util F` | Fraction of GPU memory, for example `0.92` | From the preset |
| `--host H` | Listen address. `0.0.0.0` has no authentication | `127.0.0.1` |
| `--port P` | Listen port | `8100` |
| `--gpus 0,1` | GPU indices, ascending. Sets `HIP_VISIBLE_DEVICES` and the TP size | `0,1` |
| `--tools`, `--no-tools` | Automatic tool choice | On |
| `--vision`, `--no-vision` | Image and video inputs | From the preset |
| `--offload-gb N` | KV offload to host memory, in GiB. `0` is off | `0` |
| `--tries N` | Attempts before giving up | `3` |
| `--kill-stuck` | Between attempts, remove a rank that survives `stop.sh` with `kill-stuck-rank.sh --force` (SIGKILL, behind checks of the kernel log) | Off |
| `--start-timeout S` | Seconds one attempt may take to answer | `1200` |
| `--no-probe` | Do not send the first chat request after startup | The probe is sent |
| `--foreground` | Run the server once in the foreground, with no retries, for systemd | Background |
| `--print-command` | Print the environment and the command, and start nothing | |
| `--list` | List the presets | |

Exit codes: 0 running; 2 bad call or a missing requirement; 3 not installed; 4 the driver check failed; 5 no attempt worked; 6 a previous server did not stop.

**Always passed by `serve.sh`:**

```
PATH=<venv>/bin:<ROCm>/bin:$PATH HIP_VISIBLE_DEVICES ROCM_PATH PYTORCH_ROCM_ARCH=gfx1030 SAFETENSORS_FAST_GPU=1 OMP_NUM_THREADS=4
--disable-custom-all-reduce --dtype float16 --attention-backend TRITON_ATTN --shutdown-timeout 60
--pipeline-parallel-size 1 --tensor-parallel-size <number of GPUs>
```

Each of these has a reason:

- `--disable-custom-all-reduce` keeps vLLM's own fast path off. It assumes direct access to the other card (KI-4). The host all-reduce of this fork still works.
- `--dtype float16` is required, because RDNA2 has no bf16 instructions (KI-3).
- `--attention-backend TRITON_ATTN` selects the optimized path; `ROCM_ATTN` takes an unoptimized branch on gfx1030.
- `--shutdown-timeout 60` keeps upstream from killing workers at once (KI-2).

IDs such as KI-2 refer to [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Presets

| Preset | Window | KV | `--util` | Streams | MTP by default | Vision by default | Parsers |
|---|---:|---|---:|---:|---|---|---|
| `qwen3.8-27b` | 262,144 | int4 | 0.92 | 8 | no | no | `qwen3_xml`, `qwen3` |
| `qwen3.5-9b` | 262,144 | int4 | 0.92 | 8 | no | no | `qwen3_xml`, `qwen3` |
| `qwen3.5-35b-a3b` | 131,072 | int4 | 0.97 | 4 | no | yes | `qwen3_xml`, `qwen3` |
| `qwen3.6-35b-a3b` | 262,144 | int4 | 0.97 | 8 | yes | yes | `qwen3_xml`, `qwen3` |
| `gemma-4-26b-a4b` | 262,144 | int4 | 0.97 | 16 | not available | yes | `gemma4`, `gemma4` |

Switches set by the presets:

| Preset | `PRESET_ENV` |
|---|---|
| `qwen3.8-27b`, `qwen3.5-9b` | `VLLM_ROCM_W4A8=256 VLLM_ROCM_W4A8_TILE=8 VLLM_ROCM_LMHEAD_INT8=1 VLLM_ROCM_EMBED_UVA=1 VLLM_ROCM_MIN_P_FLOOR=0.05 VLLM_ROCM_MAX_TEMPERATURE=0` |
| `qwen3.5-35b-a3b`, `gemma-4-26b-a4b` | `VLLM_ROCM_MOE_RDNA2=1 VLLM_ROCM_MAX_TEMPERATURE=0` |
| `qwen3.6-35b-a3b` | `VLLM_ROCM_MOE_RDNA2=1 VLLM_ROCM_W8A16=1 VLLM_ROCM_MAX_TEMPERATURE=0` |

The measured configuration of each row in the tables is in [PERFORMANCE.md](PERFORMANCE.md#configuration-of-each-run). The presets can differ from it. The Qwen3.6 preset allows 8 streams, as in the agent session; its table rows were measured with 4, so repeating a row takes `--streams 4`, and `--ctx 131072` or `--no-mtp` where the row says so.

## Own presets

Copy a preset under a new name in `presets/`, and the new name works everywhere. The name uses lowercase letters, digits, `.`, `_` and `-`. The file holds only `KEY=value` lines and comments. `serve.sh` reads it as data and never runs it.

| Key | Required | Meaning |
|---|---|---|
| `HF_ID` | yes | Hugging Face repository |
| `CONVERT` | | Empty, or `awq-to-ct`: `download-model.sh` converts, and the weights are in `models/PRESET-ct-int4` |
| `SERVED_NAME` | yes | Model name in the API |
| `CTX` | yes | Window in tokens |
| `KV` | yes | `int4_per_token_head`, `int8_per_token_head` or `auto` |
| `UTIL` | yes | Fraction of GPU memory |
| `MAXSEQ` | yes | Concurrent sequences |
| `MTP` | yes | `1` or `0`: on by default or not |
| `MTP_METHOD` | if `MTP=1` | Method of `--speculative-config`. Empty: `--mtp` is refused |
| `MTP_CAPTURE_SIZES` | if `MTP=1` | JSON list for `cudagraph_capture_sizes`, needed because speculation changes the graph sizes (KI-15) |
| `MTP_NEEDS_HEAD` | | `1`: `--mtp` needs `model-mtp.safetensors` in the model directory |
| `TOOL_PARSER` | yes | `--tool-call-parser` |
| `REASONING_PARSER` | yes | `--reasoning-parser` |
| `VISION_KWARGS` | yes | `1`: pass `max_pixels` to the processor when vision is on |
| `VISION_DEFAULT` | yes | `1`: vision on by default |
| `PRESET_ENV` | yes | Environment of the server: `KEY=value` entries, separated by spaces |
| `REASON_END` | | Reasoning-end phrase, see below. Absent: the shared default. Empty: off |
| `REASON_GRACE` | | Tokens after the phrase. Default 2048 |
| `BATCHTOK` | | `--max-num-batched-tokens`. Default 8192 |
| `LONGPREFILL` | | `--long-prefill-token-threshold`. Default 2048 |

`PRESET_ENV` must not set `PYTORCH_HIP_ALLOC_CONF` (KI-5). It must also not set `VLLM_ROCM_INT4_DECODE_QLEN_MAX` and `VLLM_REASONEND_GRACE`, which `serve.sh` computes. Any of these is refused with exit code 2.

A model outside this list is untested: start from the closest preset and expect to adjust. See [what the kernels support](../README.md#quantizations-and-context-lengths).

## Choosing settings

**Window and KV cache.**

- int4 is the default and holds about ×1.8 the tokens of int8. It also decodes faster on long prompts on the tested path, ×2.3–3 at 125,000 tokens.
- A window above 131,072 tokens needs int4 on these models. If a window does not fit, the start fails, and the message states the limit (KI-6).
- A window is a limit for one request. The KV cache is shared by all streams, and its size is in the log, as `GPU KV cache size`.

**Memory fraction.** Lower `--util` if the card has other users. A display on a compute card holds 0.35–0.69 GiB, which limits `--util` to about 0.978 on 16 GB.

**Streams.** More streams raise the total throughput and lower the speed of each. The measured points are in [the matrices](PERFORMANCE.md#full-matrices).

**MTP.** The gain is largest for one stream on a long prompt. Numbers on random tokens are inflated. On a real agent workload with 4 clients, Qwen3.6-35B-A3B gained 5.6 % in wall time. Two things to know:

- with MTP on, `min_p` and `logit_bias` in a request are refused (KI-17);
- a client abort during prefill can hang the workers (KI-9).

So use long client timeouts, or `--no-mtp`.

**Vision.** Vision is on by default where the preset says so. A vision check on Qwen3.5-35B-A3B passed 3 of 4 cases, with small text the weak point, and the same check passed on Qwen3.8-27B. The other models were not checked, so vision is not a verified feature of this release.

**KV offload.** `--offload-gb N` keeps evicted KV blocks in host memory. It helps when an evicted context is used again: ×27 on a repeated context of 90,000 tokens. It does nothing for a context that is read once. Under parallel sessions the transfer queue is serial, and with 4 streams one request waited for an hour (KI-25). Two crashes with a driver message happened under it, with the link not proven (KI-11). `/dev/shm` must have the space.

**More than two cards.** The fast all-reduce handles exactly two ranks. Other counts use RCCL and were not measured with `serve.sh`. Pipeline parallelism needs a manual `vllm serve` command ([example](PERFORMANCE.md#the-pipeline-rows)), and it does not work with MTP (KI-27). The three-card comparison is in [PERFORMANCE.md](PERFORMANCE.md#against-pipeline-parallelism-on-three-cards).

## Environment variables

**Of the scripts**

| Variable | Meaning | Default |
|---|---|---|
| `VLLM_RDNA2_HOME` | Install location | `~/.local/share/vllm-rdna2` |
| `ROCM_PATH` | ROCm directory | `/opt/rocm` |
| `REASONEND` | Reasoning-end phrase for one run, even if empty. Overrides the preset | |
| `PROBE_TIMEOUT` | Limit of the probe in `gpu-health.sh`, in seconds | `60` |

**Of the server.** Variables in your shell reach the server, and the values in `PRESET_ENV` take priority over them. To test a switch that the preset does not set:

```bash
AMD_SERIALIZE_KERNEL=3 ./serve.sh qwen3.8-27b
```

To change a value that the preset sets, copy the preset and edit the copy. For a unit, add `Environment=` lines. The switches added by this fork are listed in [SWITCHES.md](SWITCHES.md). Change one at a time, and do not treat a value as final after one run: the first section lists which ones can change the model output.

## Reasoning end phrase

With `thinking_token_budget` in a request, the server ends the reasoning when the budget is spent. A hard cut can leave a turn with neither an answer nor a tool call (KI-23). With a phrase set, the server adds it to the output and gives the model `REASON_GRACE` more tokens to finish by itself.

- The shared default is `Okay, I've spent enough time on this. Let me wrap up my reasoning and give the answer.` The Qwen presets use it.
- The Gemma preset has it off (`REASON_END=` empty).
- A phrase that ends with `</think>` leaves no room for the grace tokens, and `serve.sh` warns about it.

## Using the API

The server speaks the OpenAI API. The model name is the `SERVED_NAME` of the preset.

```bash
curl -s http://127.0.0.1:8100/v1/models

curl -s http://127.0.0.1:8100/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.6-35b-a3b",
  "messages": [{"role": "user", "content": "Explain tensor parallelism in two sentences."}],
  "max_tokens": 256,
  "stream": true,
  "stream_options": {"include_usage": true}
}'
```

With Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8100/v1", api_key="unused", timeout=1800)
reply = client.chat.completions.create(
    model="qwen3.6-35b-a3b",
    messages=[{"role": "user", "content": "Explain tensor parallelism in two sentences."}],
    max_tokens=256,
)
print(reply.choices[0].message.content)
```

- **Timeouts.** During prompt processing the server sends nothing, so the client timeout has to exceed the longest prefill. A prompt of 125,000 tokens takes 93–441 s, depending on the model.
- **Reasoning.** The reasoning parser moves reasoning text out of `content`. Set `max_tokens` high enough: with reasoning models a low limit can end before any answer starts.
- **Sampling defaults.** The server takes defaults for missing fields from `generation_config.json` of the model.
- **Authentication.** There is none. Keep the default `127.0.0.1`, or put your own proxy in front.
