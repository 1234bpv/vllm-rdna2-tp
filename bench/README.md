# Benchmarks and checks

Scripts that measure a running vLLM (or llama-server) instance on a two-card
RDNA2 machine, and check the collective communication they depend on.

Run them against a server that is already up. `serve.sh` starts the server; the
scripts here never start, stop or change it.

## Files

| File | What it measures |
|---|---|
| `kv-concurrency.py` | Several requests at once: VRAM and KV-cache occupancy, running and waiting requests, preemptions, time to first token, prefill and decode speed per stream, and an aggregate decode speed counted only in the window where all streams decode. One JSONL record per run. |
| `run-matrix.sh` | Runs `kv-concurrency.py` over a list of `LENGTH:STREAMS` points against a running server, and appends the records to one JSONL file. |
| `moe-speed2.py` | Aggregate decode speed of several simultaneous streams, from the server's token counter, in the window where all streams decode. Use it instead of summing per-stream speeds. |
| `needle.py` | Needle-in-a-haystack recall: four facts inserted at four positions in long filler text, four questions. Shows whether quantization spoils retrieval in long context. |
| `compat-prompts.py` | Requests with prompts of 1 to 64 tokens, and a mixed run of long decodes with tiny prompts. Checks the output length of each request. Exit 1 on any error or short output. |
| `isendrecv.py` | Two-rank `isend`/`irecv` correctness over RCCL, the path pipeline parallelism uses between stages. |
| `hostar_test.py` | Two-process correctness and latency of the host all-reduce custom op. |
| `arlat.py` | All-reduce latency at decode-sized tensors (hidden size 5120, fp16) over RCCL (the `nccl` backend name on ROCm). |

## Running

Requirements: two ROCm GPUs, PyTorch with ROCm, and vLLM built with the ROCm
custom ops. Multi-process checks are started with `torchrun`, one process per GPU:

```
torchrun --nproc-per-node 2 isendrecv.py
torchrun --nproc-per-node 2 hostar_test.py
torchrun --nproc-per-node 2 arlat.py
```

Run them with the `torchrun` of the venv, with `<venv>/bin` and `<ROCm>/bin` on `PATH`. **`torchrun` sends SIGKILL to a worker that
survives SIGTERM for 30 s**, and SIGKILL of a process with pending GPU faults can hang the driver
([KNOWN_ISSUES.md](../docs/KNOWN_ISSUES.md), KI-2). If a worker is stuck, see [TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md) first.

The release check started the two ranks by hand instead, so that nothing can send SIGKILL:

```
for rank in 0 1; do
    RANK=$rank LOCAL_RANK=$rank WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29555 HIP_VISIBLE_DEVICES=0,1 \
        timeout 300 <venv>/bin/python hostar_test.py &
done
wait
```

Start the server, then run a matrix. Take the block count from the server log line
`GPU KV cache size`:

```
bash bench/run-matrix.sh my-preset --url http://127.0.0.1:8100 --blocks 74
```

`--blocks` is the block count from that log line; `0` (the default) means not given,
so the `blocks_peak` field is null and is not compared.

`run-matrix.sh` options:

* `--points "300:1 300:4 125000:2"`: `LENGTH:STREAMS` pairs. Default: eight points
  from 300 tokens to 125000 tokens, with 1, 2 and 4 streams.
* `--output FILE`: JSONL file to append to. Default `results/local/PRESET-DATE.jsonl`.
* `--label L`: text stored in each record. Each point is labelled `L point=L<length>x<streams>`.
* `--point-timeout S`: time limit for one point in seconds (default 3000). A point that
  hits the limit or fails is reported, and the next point runs.
* `--url URL`: address of the running server (default `http://127.0.0.1:8100`).
* `--blocks N`: the block count, as described above (default `0`, not given).

The script exits 0 when all points succeed, 1 when any point failed, 2 on bad
arguments, and 3 when the server does not answer `/v1/models`.

Environment variables read by `kv-concurrency.py` and recorded in each record:
`MAXSEQ`, `SPEC_N`, `KV`, `TRASA`, `KVBYTES`, `BATCHTOK`, `TP`, `PP`, `UTIL`,
`SERVER_PIDNJATO`. `KV_PROBE_CARDS=card0,card1` restricts the VRAM reading to the
cards the server uses. `MODEL_NAME` sets the model name in `compat-prompts.py`
(default `q9`). `SPEED2_SALT` fixes the prompt salt in `moe-speed2.py`.

## Reading the records

Each `kv-concurrency.py` record has the fields `etap` (label), `prylad` (server
and environment), `umovy` (conditions), `chyslo` (measurements) and `dokaz`
(`pomylky`: the list of errors; an empty list means a clean run). The fields
most often compared are `chyslo.sukupna_shvydkist_tok_s` (aggregate decode speed),
`chyslo.blokiv_pik` (peak KV blocks in use), and `chyslo.vytisnen_riznytsia`
(growth of the preemption counter during the run).

The record field names are kept in the original form on purpose: published
records use them. The English command-line names (`--base-url`, `--streams`,
`--length`, `--blocks`, `--output`, `--label`, `--seed`, `--request-timeout`) are
aliases of the original Ukrainian names (`--baza`, `--potokiv`, `--dovzhyna`,
`--blokiv`, `--vyhid`, `--mitka`, `--zerno`, `--mezha-zapytu-s`). Both work.

Prompt texts in `compat-prompts.py`, `needle.py` and `moe-speed2.py` are in
Ukrainian on purpose. They are the measured input: changing them changes the
token counts and the result.

## Why not "words per second"

A throughput measured in words per second is not comparable across prompts.
The same number of words tokenizes into a different number of tokens depending
on the text. On the test machine a prefill measured in words per second first
suggested a sixfold drop. The drop was an artifact of the unit: the random words
tokenized into very different numbers of tokens. The measurement was then repeated
in tokens. Report speed in tokens per second, with the token counts taken from the
server's `usage` field (`usage.prompt_tokens`, `usage.completion_tokens`).

## Why one RCCL run is noise

Collective failures in RCCL are not stable between runs. The known-defects note
records a failure rate near 50/50, which is settled at communicator initialization
and stays stable for the rest of the process. In the recorded tests a single run
showed 0 errors in 90 checks per operation, and a second run gave 90 of 90 wrong
`all_gather` results on rank 0. A single run that shows an improvement, or a
regression, often does not reproduce: several "found fixes" did not hold on
repetition. Any comparison of RCCL settings needs at least 6 to 8 runs, and the
conclusion should rest on the runs that agree.

## Status of these scripts

The scripts were checked for syntax: `bash -n` for the shell scripts and a Python
`ast` parse for the Python files. The measurement logic of `kv-concurrency.py` was
compared with the original against a stub server that returns OpenAI-style streaming
responses, in both vLLM and llama-server modes. The records matched in structure and
in every deterministic value; the only difference is the translated text of error
messages. The multi-process RCCL checks and the measurements on real hardware have not
been re-run as part of the release.
