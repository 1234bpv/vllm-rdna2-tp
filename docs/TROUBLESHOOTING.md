# Troubleshooting

IDs such as KI-1 refer to [KNOWN_ISSUES.md](KNOWN_ISSUES.md). The commands below all have a time limit. Use them first:

```bash
timeout 120 scripts/doctor.sh                     # PASS / WARN / FAIL for each check
timeout 60 scripts/gpu-health.sh; echo $?         # 0: GPU usable; 1: the driver needs a reboot
timeout 5 pgrep -af "[v]llm serve|[V]LLM::"       # vLLM processes still alive
timeout 5 ss -lnt | grep ':8100 '                 # the default port still bound
timeout 10 rocm-smi --showmeminfo vram            # VRAM in use per card
timeout 10 rocm-smi --showpids                    # who holds the GPUs
timeout 30 journalctl -k --since "10 min ago" --no-pager | grep -iE "page fault|hogged CPU|GPU reset|ring .* timeout|Fatal error during GPU init"
```

Server logs are in `~/.local/share/vllm-rdna2/logs/serve-<preset>-<time>.log`.

## The server does not start

**Exit codes of `serve.sh`**

| Code | Meaning | Next step |
|---:|---|---|
| 2 | Bad option, unknown preset, or a preset requirement is missing, for example the MTP head or the converted weights | Read the message |
| 3 | Not installed | `./install.sh` |
| 4 | The driver check failed | [Reboot](#the-machine-needs-a-reboot) |
| 5 | No attempt produced a working server | Read the log, then see below |
| 6 | A previous server did not stop. Nothing new was started | [The process does not exit](#the-process-does-not-exit) |

**Messages in the log**

| Message | Meaning | Do |
|---|---|---|
| `hipErrorIllegalAddress`, `illegal memory access` | The random fault, KI-1 | `serve.sh` already retried. Run it again, or try `AMD_SERIALIZE_KERNEL=3 ./serve.sh ...` |
| `To serve at least one request with the model's max seq len` | The window does not fit, KI-6 | Lower `--ctx` |
| `Insufficient space in /dev/shm` | KI-7 | `df -h /dev/shm`. Remove `/dev/shm/vllm_offload_*` only when no server is running |
| `hipErrorOutOfMemory` on PyNccl initialization | `expandable_segments`, KI-5 | Unset `PYTORCH_HIP_ALLOC_CONF` |
| `LLVM ERROR: Cannot select intrinsic llvm.amdgcn.fdot2.bf16.bf16` | KI-3 | `--dtype float16` |
| `There is no module or parameter named 'fc.weight'` | KI-18 | Add the MTP modules to `quantization_config.ignore` |
| `AssertionError` in `offloading/config.py` | KI-26 | Do not set `block_size` |
| `Permission denied` on `/dev/kfd` | The user is not in the `render` and `video` groups | `sudo usermod -aG render,video "$USER"`, then log in again |

**The first start takes minutes.** Compilation, for example one cold `torch.compile` of 119 s, happens once; later starts reuse the cache. The `.hip` kernels are compiled at the first use. That needs `hipcc` and an editable install.

**Do not:**

- start a second server while the first holds VRAM;
- reverse the GPU order (KI-8);
- draw a conclusion from one failed start: they are random;
- force-kill a half-started process (KI-2).

## The first request hangs

| Sign | Meaning |
|---|---|
| No output, and the log shows growing KV cache use | Normal prefill. A prompt of 125,000 tokens takes 93–441 s, depending on the model |
| GPUs at 99 %, workers at about 300 % CPU, the counters do not move | MTP hang after a client abort, KI-9 |
| `TimeoutError: RPC call to sample_tokens timed out` | Reversed GPU order, KI-8 |
| Kernel log `page fault … address 0x0 … UTCL2 client TCP` | The random fault, KI-1. A rank has died |

To check the counters, run this twice, a few seconds apart:

```bash
timeout 5 curl -s http://127.0.0.1:8100/metrics | grep -E "generation_tokens_total|spec_decode_num_drafts_total"
```

To check the GPU order of the process:

```bash
timeout 5 tr '\0' '\n' < /proc/<PID>/environ | grep HIP_VISIBLE
```

Keep client timeouts longer than the longest prefill. With MTP on, do not abort requests during prefill. If clients abort often, use `--no-mtp`. If a rank stays, run `./stop.sh`, wait, and continue with [The process does not exit](#the-process-does-not-exit).

## The output is garbage or repeats

Check first:

1. Send the same prompt twice at temperature 0. The outputs must be identical.
2. Compare with the `smoke-test.py` output, and check that reasoning text is separated from the answer in the API response.
3. Judge quality on many samples. One bad sample is not evidence.

| Sign | Likely cause | Do |
|---|---|---|
| Fragments in non-English text at temperature 1.0 | Quantization of the weights, KI-21 | Weights with group 32, or keep `min_p` at 0.05 without MTP |
| Loops inside long reasoning | A temperature ceiling, KI-22 | `VLLM_ROCM_MAX_TEMPERATURE=0` |
| Reasoning text inside the answer | The client shows the reasoning field, or an argument after `--` replaced the parser | `serve.sh` always passes the reasoning parser of the preset. Compare with `./serve.sh PRESET --print-command` |
| Stops after 20–100 tokens, or one character per line, with MTP | A build without the MTP fix | Check `git -C ~/.local/share/vllm-rdna2/src/vllm diff --stat`. On 0.1.0 please report it |
| Different answers at temperature 0 | On the MoE kernel, `VLLM_ROCM_MOE_RDNA2_DETERMINISTIC=0`; otherwise worth reporting | Keep the default of 1 |
| Random text from the first token | `VLLM_ROCM_COLLECTIVES_VIA_ALLREDUCE=0`, KI-12 | Keep the default of 1 |

## Decode is very slow

Read the settings of the running server first:

```bash
PID=$(pgrep -f "[v]llm serve" | head -1)
timeout 5 tr '\0' ' ' < /proc/$PID/cmdline; echo
timeout 5 tr '\0' '\n' < /proc/$PID/environ | grep -E "^(VLLM|HIP|AMD)_"
```

Compare with `./serve.sh PRESET --print-command`, and then:

| Check | Why |
|---|---|
| Is the server on the expected GPU clock? `rocm-smi --showclocks`. The published numbers are at 1600 MHz | A tuning daemon can change clocks. One such daemon cut generation by 26 % with a memory clock above 1000 MHz on the reference machine. Not confirmed on other cards |
| `rocm-smi --showtemp`: junction temperature | Consumer cards reduce the clock under sustained load |
| `cat /proc/sys/kernel/numa_balancing`. `sudo sysctl -w kernel.numa_balancing=0` if you accept the change | RCCL names it as a source of variability |
| MTP with many streams | KI-16 |
| Do you measure with `usage` tokens over one shared time window? | Stream fragments and per-thread timers mislead. Use `bench/kv-concurrency.py` |
| Are you profiling under full CUDA graphs? | Profiles were up to four times too large in that mode. Confirm with an A/B of a switch |

Leave the attention backend at `TRITON_ATTN`, as `serve.sh` passes it: `ROCM_ATTN` takes an unoptimized branch on gfx1030. Do not trust one run of a benchmark.

## Out of memory

```bash
timeout 10 rocm-smi --showmeminfo vram     # before a start: another process may hold VRAM
timeout 10 rocm-smi --showpids
df -h /dev/shm; du -sh /dev/shm/vllm_*
```

- Lower `--util`, `--ctx` or `--streams`. A display on a compute card holds 0.35–0.69 GiB.
- A smaller prefill chunk helps with prompt logprobs, KI-24.
- Do not use `expandable_segments` (KI-5).

## The process does not exit

`./stop.sh` sends only SIGTERM, and it returns 1 if a process survives.

1. Wait, then try again with a longer limit: `timeout 150 ./stop.sh --timeout 120`.
2. Run `timeout 60 scripts/gpu-health.sh`.
3. For one stuck process: `scripts/kill-stuck-rank.sh PID`. It sends SIGTERM only if the PID is a vLLM process, no rank is alive for an engine process, and the driver check passes. It sends no SIGKILL unless you pass `--force`. Exit code 0: the process is gone and the driver is healthy. Exit code 1: the driver check failed, or the process is still alive after 30 s. Exit code 2: nothing was signalled, for example because the PID is not a vLLM process.
   With `--force`, a rank that survives SIGTERM for 30 s gets SIGKILL if the kernel log can be read and has no page fault from this PID or on the card of the rank since it started. Otherwise the rank stays and the script exits with 1. `./serve.sh --kill-stuck` does this between attempts, so a failed start can be retried without a reboot. SIGKILL can still freeze a machine with unfinished GPU page faults (KI-2). This path ran on a real GPU process held with SIGSTOP, and has not yet met a rank stuck after a real fault.
4. If processes remain, reboot. **Do not use `kill -9`** (KI-2). A rank in state `R` that ignores SIGTERM after KI-1 is this case: its main thread is inside a GPU wait, so the signal is never handled. Waiting did not help in the observed cases (about 500 s and 280 s, until the reboot).
5. Do not start a new server while old processes hold VRAM. `serve.sh` refuses with code 6.

## The machine needs a reboot

Signs:

- `scripts/gpu-health.sh` exits with 1;
- kernel log `workqueue: kfd_process_wq_release [amdgpu] hogged CPU for >10000us`, then a freeze (KI-2);
- kernel log `GPU reset`, `ring … timeout` or `Fatal error during GPU init`;
- a rank that does not exit after a page fault;
- `rocminfo` reports `HSA_STATUS_ERROR_OUT_OF_RESOURCES` and `power/runtime_status` of a card is `error` (KI-38).

Save the matching kernel log lines before rebooting. After a driver failure, do not start again until `scripts/gpu-health.sh` returns 0: `serve.sh` refuses with code 4. Changing kernel options such as `iommu=pt` is not a fix: RCCL warns about it, but a check found it unrelated to these failures.

## Reporting a problem

Please include the output of `scripts/doctor.sh`, the card models, the command from `serve.sh --print-command`, the error lines from the server log and the kernel log, and how often the failure occurs. Remember that a single run proves little on this platform.
