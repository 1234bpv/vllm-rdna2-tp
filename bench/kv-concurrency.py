#!/usr/bin/env python3
"""Concurrency measurement: several requests at once, to tell queueing from real parallelism.

The single-request probe (kv-probe.py, not part of this release) measures one request;
the record schema is the same. This file is self-contained on purpose: the small helpers
are duplicated so that the tool stays readable on its own.

The main caveat: a snapshot every 0.5 s misses the peak on short requests. So this tool
samples every 0.2 s and records not only VRAM and kv_cache_usage_perc but also
vllm:num_requests_running and vllm:num_requests_waiting. Without those, "4 streams" could
in fact run one after another in a queue, and the measurement would not show it. Each
stream reads its answer as an SSE stream (stream=true) to capture the time to the first
token, not only the total duration.

    python3 kv-concurrency.py --base-url http://127.0.0.1:8100 \\
        --streams 4 --length 125000 --blocks 74 \\
        --output kv-concurrency.jsonl

``--engine llamacpp`` runs the same measurement against llama-server (engine comparison).
Instrument differences: the request occupancy comes from ``llamacpp:requests_processing``
and ``requests_deferred``. llama.cpp has no KV-cache occupancy metric and no preemption
counter (it does not preempt: a request that does not fit in a slot fails, and the stream
catches that error), so those fields are None rather than a failure. Pass ``--blocks 0``
with that engine; the value is not used.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime

SETTLE_TOLERANCE_BYTES = 10 * 1024 * 1024   # two VRAM readings closer than this count as settled
ROCM_SMI_TIMEOUT_S = 30
HTTP_TIMEOUT_S = 15
MIN_TOKEN_ID = 1000
MAX_TOKEN_ID = 200000
SAMPLE_INTERVAL_S = 0.2


def _env_int(name):
    """Integer from an environment variable, or None. An empty string is None too:
    otherwise `KVBYTES=` would silently become zero and the record would look valid."""
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def run_limited(command, limit):
    """Run a command with a time limit. Without one, silence cannot be told from work.

    On expiry the child gets SIGTERM only: subprocess.run would send SIGKILL and then wait
    without a limit, so a tool stuck in the driver would hang the probe.
    """
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        out, _ = child.communicate(timeout=limit)
    except subprocess.TimeoutExpired:
        child.terminate()
        try:
            child.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return None
    return out


def http_request(url, payload=None, timeout=HTTP_TIMEOUT_S):
    """GET (payload=None) or POST JSON, always with a time limit. The body is read whole."""
    try:
        if payload is None:
            req = urllib.request.Request(url, method="GET")
        else:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"), method="POST",
                headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except Exception as exc:
        print(f"network call to {url} failed: {exc}", file=sys.stderr)
        return None


def vram_bytes():
    """Used VRAM per card, from the driver counters."""
    out = run_limited(["rocm-smi", "--showmeminfo", "vram", "--csv"], ROCM_SMI_TIMEOUT_S)
    if not out:
        return None
    cards = {}
    # KV_PROBE_CARDS=card0,card1 restricts the reading to the server's cards. Otherwise a
    # micro-benchmark on a free third card keeps the driver from ever settling.
    only = {c for c in os.environ.get("KV_PROBE_CARDS", "").split(",") if c}
    for line in out.splitlines():
        parts = line.strip().split(",")
        if only and parts[0] not in only:
            continue
        if len(parts) >= 3 and parts[0].startswith("card"):
            try:
                cards[parts[0]] = int(parts[2])
            except ValueError:
                return None
    return cards or None


def wait_for_settle(pause, attempts):
    """Two readings in a row must agree; otherwise the state has not settled yet."""
    previous = vram_bytes()
    if previous is None:
        return None, "rocm-smi did not respond"
    for attempt in range(1, attempts + 1):
        time.sleep(pause)
        current = vram_bytes()
        if current is None:
            return None, "rocm-smi did not respond"
        if set(current) != set(previous):
            return None, "list of cards changed between readings"
        diff = {k: abs(current[k] - previous[k]) for k in current}
        if all(v <= SETTLE_TOLERANCE_BYTES for v in diff.values()):
            return current, None
        print(f"  reading {attempt}: not settled yet, difference "
              + ", ".join(f"{k} {v / 2**20:.1f} MiB" for k, v in diff.items()),
              file=sys.stderr)
        previous = current
    return None, f"driver did not settle within {attempts} readings {pause} s apart"


def weights_and_window(base):
    """The server itself is the only reliable source of the weights format and the window,
    not the launch line."""
    text = http_request(f"{base}/v1/models")
    if text is None:
        return None, None
    try:
        data = json.loads(text)["data"][0]
        if ENGINE == "vllm":
            return data["root"], data.get("max_model_len")
        # llama-server: the GGUF path is in id, the working window (all slots together) in /props.
        window = None
        props = http_request(f"{base}/props")
        if props:
            window = json.loads(props).get("default_generation_settings", {}).get("n_ctx")
        return data["id"], window
    except (TypeError, ValueError, KeyError, IndexError):
        return None, None


def metric_value(text, name):
    """One gauge or counter value from /metrics output. A missing line is None, not zero."""
    for line in text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            try:
                return float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                return None
    return None


METRIC_NAMES_BY_ENGINE = {
    "vllm": {
        "kv_cache_usage_perc": "vllm:kv_cache_usage_perc",
        "num_preemptions_total": "vllm:num_preemptions_total",
        "num_requests_running": "vllm:num_requests_running",
        "num_requests_waiting": "vllm:num_requests_waiting",
    },
    # None: the engine has no such metric at all (as opposed to "could not be read").
    "llamacpp": {
        "kv_cache_usage_perc": None,
        "num_preemptions_total": None,
        "num_requests_running": "llamacpp:requests_processing",
        "num_requests_waiting": "llamacpp:requests_deferred",
    },
}
ENGINE = "vllm"            # set by main() from --engine
METRIC_NAMES = METRIC_NAMES_BY_ENGINE[ENGINE]


def read_metrics(base):
    """Dict {field: number}, or {field: None} where the line is absent from the output.

    ⚠️ A missing line is None, NOT 0.0. Once vllm:gpu_cache_usage_perc (which does not exist)
    was confused with vllm:kv_cache_usage_perc (the real name), and the wrong name silently
    gave zero. The other metrics read here carry the same risk, so each one is checked.
    """
    text = http_request(f"{base}/metrics")
    if not text:
        return {k: None for k in METRIC_NAMES}
    return {k: (metric_value(text, name) if name else None) for k, name in METRIC_NAMES.items()}


class Sampler(threading.Thread):
    """While requests are in flight, reads VRAM and engine metrics every SAMPLE_INTERVAL_S seconds."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self._stop_event = threading.Event()
        self.rows = []
        self.error = None

    def run(self):
        start = time.monotonic()
        while not self._stop_event.is_set():
            cards = vram_bytes()
            if cards is None and self.error is None:
                self.error = "rocm-smi returned empty output during the requests"
            metrics = read_metrics(self.base)
            self.rows.append({
                "chas_s": round(time.monotonic() - start, 3),
                "vram_bajt": cards,
                "kv_cache_usage_perc": metrics["kv_cache_usage_perc"],
                "num_requests_running": metrics["num_requests_running"],
                "num_requests_waiting": metrics["num_requests_waiting"],
                "num_preemptions_total": metrics["num_preemptions_total"],
            })
            self._stop_event.wait(SAMPLE_INTERVAL_S)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=10)

    def peak_vram(self):
        peak = {}
        for row in self.rows:
            if not row["vram_bajt"]:
                continue
            for k, v in row["vram_bajt"].items():
                if v > peak.get(k, -1):
                    peak[k] = v
        return peak or None

    def peak(self, field):
        values = [r[field] for r in self.rows if r[field] is not None]
        return max(values) if values else None


def build_prompt(length, seed):
    """A prompt of random token IDs: no text, so the prefix cache cannot help."""
    rng = random.Random(seed)
    return [rng.randint(MIN_TOKEN_ID, MAX_TOKEN_ID) for _ in range(length)]


def run_stream(base, index, prompt, seed, max_tokens, expected_length, timeout_s, results,
               shared_start, stamps_by_stream):
    """Sends one streaming request (SSE) to catch the time to the first token.

    ``stream_options.include_usage`` asks the server to add usage to the last chunk.
    Without it, the prompt length in streaming mode cannot be checked.

    ⚠️ Fixed defect: an earlier version counted a token as "first" only when
    ``choices[0].text`` was a non-empty string. The vLLM detokenizer legitimately returns
    empty text for a token that is part of a multi-byte UTF-8 sequence (Cyrillic!) or a
    special character that has not finished its sequence, while the token is still
    generated. On raw requests (build_prompt gives random token IDs, not text) the model
    often lands in exactly that situation, and at temperature=0 it could keep empty text
    to the end of the 32-token budget. That is where a null with status 200 and a non-zero
    completion_tokens came from. For vLLM the count is taken from ``token_ids``, regardless
    of text and of how many tokens a chunk holds. Time is the client's receive time, not
    the GPU kernel's execution time; tokens within one chunk share the same time.
    """
    body = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        # The prompt is built from random token IDs, and the model reaches the end-of-sequence
        # token almost at once: in the 19.09 run three of four streams produced 1, 2 and 6
        # tokens instead of 200, so there was no decode speed to measure. Both fields are
        # needed: ignore_eos does not stop at EOS, min_tokens keeps any other stop rule off.
        "ignore_eos": True,
        "min_tokens": max_tokens,
    }
    if ENGINE == "vllm":
        body["return_token_ids"] = True
        body["stream_interval"] = 1
    t_start = time.monotonic()
    ttft_s = None
    prompt_tokens = None
    completion_tokens = None
    status_code = None
    error = None
    got_any = False
    stamps = []          # relative to shared_start, so that streams can be aligned
    fragments = []

    try:
        req = urllib.request.Request(
            f"{base}/v1/completions", data=json.dumps(body).encode("utf-8"),
            method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status_code = resp.getcode()
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                got_any = True
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    frag = json.loads(payload)
                except ValueError:
                    continue
                if frag.get("choices"):
                    now = time.monotonic()
                    choice = frag["choices"][0]
                    if ENGINE == "vllm":
                        ids = choice.get("token_ids")
                        if not isinstance(ids, list):
                            raise ValueError("no token_ids in the vLLM response")
                        n_tokens = len(ids)
                    else:
                        n_tokens = 1
                    stamp = round(now - shared_start, 6)
                    stamps.extend([stamp] * n_tokens)
                    fragments.append([stamp, n_tokens])
                    if n_tokens and ttft_s is None:
                        ttft_s = round(now - t_start, 3)
                usage = frag.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
    except Exception as exc:
        error = f"stream {index}: {exc}"

    duration_s = round(time.monotonic() - t_start, 3)

    if error is None and not got_any:
        error = f"stream {index}: empty server response"
    if error is None and status_code != 200:
        error = f"stream {index}: response status {status_code}"
    if error is None and prompt_tokens != expected_length:
        error = f"stream {index}: prompt length {prompt_tokens} instead of {expected_length}"
    if error is None and completion_tokens != max_tokens:
        error = f"stream {index}: produced {completion_tokens} tokens instead of {max_tokens}"
    if error is None and ENGINE == "vllm" and len(stamps) != completion_tokens:
        error = f"stream {index}: token_ids count does not match usage"

    # Occasionally the server adds an empty "finish" choice after the last real token (a
    # finish_reason signal with no new token). If there are more stamps than usage confirms,
    # the tail is trimmed so that last_token_s is not shifted onto it.
    if completion_tokens is not None and len(stamps) > completion_tokens:
        stamps = stamps[:completion_tokens]
    stamps_by_stream[index] = stamps

    first_token_s = stamps[0] if stamps else None
    last_token_s = stamps[-1] if stamps else None

    prefill_tok_s = None
    if prompt_tokens is not None and first_token_s is not None and first_token_s > 0:
        prefill_tok_s = round(prompt_tokens / first_token_s, 3)

    decode_tok_s = None
    if (completion_tokens is not None and completion_tokens > 1
            and first_token_s is not None and last_token_s is not None
            and (last_token_s - first_token_s) > 0):
        first_batch = (sum(p == first_token_s for p in stamps)
                       if ENGINE == "vllm" else 1)
        decode_tok_s = round(
            (completion_tokens - first_batch) / (last_token_s - first_token_s), 3)

    results[index] = {
        "nomer": index,
        "zerno": seed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "kod_vidpovidi": status_code,
        "chas_do_pershoho_tokena_s": ttft_s,
        "tryvalist_s": duration_s,
        "pomylka": error,
        "tokeniv_promtu": prompt_tokens,
        "tokeniv_vyvodu": completion_tokens,
        "chas_pershoho_tokena_s": first_token_s,
        "chas_ostannoho_tokena_s": last_token_s,
        "shvydkist_prefilu_tok_s": prefill_tok_s,
        "shvydkist_dekodu_tok_s": decode_tok_s,
        "fragmenty_tokeniv": fragments,
    }


def aggregate_decode_speed(results, stamps_by_stream):
    """Aggregate decode speed, counted ONLY in the window where ALL streams are decoding.

    Summing per-stream speeds overstates the result: while one stream is still in prefill,
    the others are already "decoding", and the other stream's prefill time gets mixed into
    their denominators (see the measurement notes: "serialization looks like perfect
    scaling"). So the window runs from the latest first token to the earliest last token,
    and only the token stamps inside it are counted.

    Returns (aggregate_tok_s, window_s, note). ``note`` is None on an exact result. On
    failure it says why there is no speed; on an estimate it says that the value is one.
    """
    if any(not r or r.get("pomylka") or r.get("chas_pershoho_tokena_s") is None
           or r.get("chas_ostannoho_tokena_s") is None for r in results):
        return None, None, "not all streams finished correctly"
    first_times = [r["chas_pershoho_tokena_s"] for r in results
                   if r and r.get("chas_pershoho_tokena_s") is not None]
    last_times = [r["chas_ostannoho_tokena_s"] for r in results
                  if r and r.get("chas_ostannoho_tokena_s") is not None]
    if not first_times or not last_times:
        return None, None, "no stream produced a token timestamp"

    window_start = max(first_times)
    window_end = min(last_times)
    window_s = round(window_end - window_start, 3)

    if window_s <= 0:
        return None, window_s, "streams did not overlap"

    # Two methods, and the choice between them is a matter of instrument validity.
    #
    # For vLLM every token_id carries the receive time of its chunk. Chunks can merge even
    # without speculation. Token counts are exact; the time stays on the client side. For
    # llama.cpp without token_ids the older estimate by window share is kept, for the case
    # with fewer chunks.
    #
    # So: if EVERY stream has exactly as many stamps as tokens, the stamps are counted.
    # Otherwise each stream's share of its own decode window that falls inside the shared
    # window is taken. That is an estimate, exact only at constant speed within a stream,
    # so the record marks it separately.
    exact = True
    for r, stamps in zip(results, stamps_by_stream):
        if not r or not stamps:
            continue
        if r.get("tokeniv_vyvodu") != len(stamps):
            exact = False
            break

    tokens_in_window = 0.0
    if exact:
        for stamps in stamps_by_stream:
            if not stamps:
                continue
            tokens_in_window += sum(1 for p in stamps if window_start < p <= window_end)
    else:
        for r in results:
            if not r:
                continue
            first = r.get("chas_pershoho_tokena_s")
            last = r.get("chas_ostannoho_tokena_s")
            produced = r.get("tokeniv_vyvodu")
            if first is None or last is None or not produced or produced < 2:
                continue
            own_window = last - first
            if own_window <= 0:
                continue
            overlap = min(window_end, last) - max(window_start, first)
            if overlap > 0:
                tokens_in_window += (produced - 1) * overlap / own_window

    note = None if exact else "estimate: chunk count differs from token count"
    return round(tokens_in_window / window_s, 3), window_s, note


def instrument(weights, window):
    return {
        "rushij": ENGINE,
        "vahy": weights,
        "vikno": window,
        "maxseq": _env_int("MAXSEQ"),
        "spec_n": _env_int("SPEC_N"),
        "kv": os.environ.get("KV") or None,
        "trasa": os.environ.get("TRASA", "0") == "1",
        "prapirci": {
            k: v for k, v in (
                ("kv_cache_memory_bytes", _env_int("KVBYTES")),
                ("batchtok", _env_int("BATCHTOK")),
                ("tp", _env_int("TP")),
                ("pp", _env_int("PP")),
                ("util", os.environ.get("UTIL") or None),
                ("token_timing", "token_ids_v2" if ENGINE == "vllm" else "fragments_v2"),
            ) if v is not None
        },
        "server_pidnjato": os.environ.get("SERVER_PIDNJATO") or None,
    }


def conditions(args, base_seed):
    return {
        "potokiv": args.streams,
        "promt_tokeniv_zadano": args.length,
        "max_tokens": args.max_tokens,
        "blokiv_zagalom": args.blocks,
        "zerno_bazove": base_seed,
    }


def write_failure(args, error, weights=None, window=None, base_seed=None):
    """Early exits are written to the JSONL too. Otherwise a failure shows up only in the log."""
    record = {
        "seriia": "kv-concurrency",
        "etap": args.label or f"potokiv-{args.streams}-dovzhyna-{args.length}",
        "chas": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prylad": instrument(weights, window),
        "umovy": conditions(args, base_seed),
        "chyslo": {},
        "dokaz": {"pomylky": [error]},
    }
    print(json.dumps(record, ensure_ascii=False, indent=2))
    with open(args.output, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"written to {args.output}")


def run_measurement(args):
    base = args.base
    errors = []

    weights, window = weights_and_window(base)
    if weights is None:
        print("server did not name the weights; the measurement is not valid, stopping",
              file=sys.stderr)
        write_failure(args, "server did not name the weights")
        return 1

    base_seed = args.seed if args.seed is not None else int(time.time() * 1000)
    prompts = [build_prompt(args.length, base_seed + i) for i in range(args.streams)]

    print(f"waiting for the driver to settle BEFORE the requests "
          f"(up to {args.attempts} readings, {args.pause} s apart)")
    vram_before, reason = wait_for_settle(args.pause, args.attempts)
    if vram_before is None:
        print(f"FAILED to settle before the requests: {reason}", file=sys.stderr)
        write_failure(args, f"settling before the requests: {reason}", weights, window, base_seed)
        return 1

    metrics_before = read_metrics(base)
    if metrics_before["num_preemptions_total"] is None and METRIC_NAMES["num_preemptions_total"]:
        errors.append("could not read vllm:num_preemptions_total before the requests")

    sampler = Sampler(base)
    sampler.start()

    results = [None] * args.streams
    stamps_by_stream = [None] * args.streams
    threads = []
    total_start = time.monotonic()
    for i in range(args.streams):
        threads.append(threading.Thread(
            target=run_stream,
            args=(base, i, prompts[i], base_seed + i, args.max_tokens,
                  args.length, args.timeout_s, results, total_start, stamps_by_stream)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=args.timeout_s + 60)
    total_s = round(time.monotonic() - total_start, 3)

    sampler.stop()

    for i, thread in enumerate(threads):
        if thread.is_alive():
            errors.append(f"stream {i}: did not finish within {args.timeout_s + 60} s")
        if results[i] is None:
            errors.append(f"stream {i}: no result, the stream did not finish writing one")
            results[i] = {"nomer": i, "zerno": base_seed + i, "prompt_tokens": None,
                          "completion_tokens": None, "kod_vidpovidi": None,
                          "chas_do_pershoho_tokena_s": None, "tryvalist_s": None,
                          "pomylka": "no result"}
        elif results[i]["pomylka"]:
            errors.append(results[i]["pomylka"])

    aggregate, window_s, aggregate_note = aggregate_decode_speed(results, stamps_by_stream)

    if sampler.error:
        errors.append(sampler.error)
    if not sampler.rows:
        errors.append("the sampler recorded no rows during the requests")
    for field in ("kv_cache_usage_perc", "num_requests_running", "num_requests_waiting"):
        if METRIC_NAMES[field] and sampler.peak(field) is None:
            errors.append(f"could not read {METRIC_NAMES[field]} during the requests")

    print(f"waiting for the driver to settle AFTER the requests "
          f"(up to {args.attempts} readings, {args.pause} s apart)")
    vram_after, reason = wait_for_settle(args.pause, args.attempts)
    if vram_after is None:
        errors.append(f"settling after the requests: {reason}")

    # ⚠️ The cumulative counter vllm:num_preemptions_total is the MAIN evidence of a memory
    # shortage; the peak kv_cache_usage_perc below is a secondary number only. On the measured
    # 16,000-token runs a snapshot every 0.2-0.5 s missed the moment of shortage: the block
    # peak showed "all fine" while the preemption counter grew by one. So the growth of the
    # counter, not the peak utilization, turns the record into a failure below.
    metrics_after = read_metrics(base)
    if not METRIC_NAMES["num_preemptions_total"]:
        preemption_delta = None      # the engine does not preempt (see --engine)
    elif metrics_after["num_preemptions_total"] is None:
        errors.append("could not read vllm:num_preemptions_total after the requests")
        preemption_delta = None
    else:
        preemption_delta = None
        if metrics_before["num_preemptions_total"] is not None:
            preemption_delta = (metrics_after["num_preemptions_total"]
                                - metrics_before["num_preemptions_total"])
            if preemption_delta > 0:
                errors.append(
                    f"preemptions grew by {preemption_delta:.0f}: memory was not enough for "
                    f"{args.streams} streams (main evidence; the peak cache utilization "
                    f"may not have shown it)")

    # Secondary number. The peak only sees what a sampler step happened to hit, and by
    # itself does NOT cause a failure (see the comment above).
    kv_peak = sampler.peak("kv_cache_usage_perc")
    # The metric's denominator is (N-1), not N: one block of the pool is the null block and
    # is never handed to a request (kv_cache_utils.py:1345-1347). Checked on six ramps: a
    # percentage over (N-1) gives an integer number of blocks, over N it does not.
    # --blocks 0 means "not given": the field stays None instead of a negative number.
    blocks_peak = (round(kv_peak * (args.blocks - 1), 3)
                   if kv_peak is not None and args.blocks > 0 else None)

    record = {
        "seriia": "kv-concurrency",
        "etap": args.label or f"potokiv-{args.streams}-dovzhyna-{args.length}",
        "chas": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prylad": instrument(weights, window),
        "umovy": conditions(args, base_seed),
        "chyslo": {
            "potoky": results,
            "vram_do_bajt": vram_before,
            "vram_pik_bajt": sampler.peak_vram(),
            "vram_pislja_bajt": vram_after,
            "kv_cache_usage_perc_pik": kv_peak,
            "blokiv_pik": blocks_peak,
            "num_requests_running_pik": sampler.peak("num_requests_running"),
            "num_requests_waiting_pik": sampler.peak("num_requests_waiting"),
            "vytisnen_riznytsia": preemption_delta,
            "tryvalist_zagalna_s": total_s,
            "sukupna_shvydkist_tok_s": aggregate,
            "vikno_spilnoho_dekodu_s": window_s,
            "prychyna_vidsutnosti_sukupnoi_shvydkosti": aggregate_note,
            "riad_sposterezhen": sampler.rows,
        },
        "dokaz": {"pomylky": errors},
    }

    for r in results:
        print(f"  stream {r['nomer']}: prefill {r.get('shvydkist_prefilu_tok_s')} tok/s, "
              f"decode {r.get('shvydkist_dekodu_tok_s')} tok/s "
              f"(prompt {r.get('tokeniv_promtu')} tokens, output {r.get('tokeniv_vyvodu')} tokens)")
    if aggregate is None:
        print(f"  aggregate decode speed: unavailable: {aggregate_note}")
    else:
        note = f" [{aggregate_note}]" if aggregate_note else ""
        print(f"  aggregate decode speed: {aggregate} tok/s "
              f"(shared decode window {window_s} s){note}")

    print(json.dumps({k: v for k, v in record.items() if k != "chyslo"}
                     | {"chyslo": {k: v for k, v in record["chyslo"].items()
                                    if k != "riad_sposterezhen"}
                                    | {"riad_sposterezhen_dovzhyna": len(sampler.rows)}},
                     ensure_ascii=False, indent=2))
    with open(args.output, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"written to {args.output} (full observation series: {len(sampler.rows)} rows)")

    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", "--baza", dest="base", required=True,
                        help="base URL of the server under test")
    parser.add_argument("--streams", "--potokiv", dest="streams", type=int, required=True,
                        help="how many requests to send at the same time")
    parser.add_argument("--length", "--dovzhyna", dest="length", type=int, required=True,
                        help="length of ONE prompt in tokens (the same in every stream)")
    parser.add_argument("--blocks", "--blokiv", dest="blocks", type=int, required=True,
                        help="block count from the server log line 'GPU KV cache size'; "
                             "0 means not given (blocks_peak is then null and not compared); "
                             "the script normalizes by (N-1) itself")
    parser.add_argument("--output", "--vyhid", dest="output", required=True,
                        help="JSONL file to append the record to")
    parser.add_argument("--max-tokens", type=int, default=32, dest="max_tokens")
    parser.add_argument("--label", "--mitka", dest="label", default=None,
                        help="free-text label stored in the record")
    parser.add_argument("--pauza", dest="pause", type=float, default=10.0,
                        help="pause between settling readings, s")
    parser.add_argument("--sprob", dest="attempts", type=int, default=6,
                        help="how many settling readings to try")
    parser.add_argument("--seed", "--zerno", dest="seed", type=int, default=None,
                        help="base seed; stream i uses seed+i (default: from the clock)")
    parser.add_argument("--request-timeout", "--mezha-zapytu-s", dest="timeout_s", type=float,
                        default=1800.0,
                        help="time limit for one streaming request, s (a long prompt takes minutes)")
    parser.add_argument("--engine", "--rushij", dest="engine", choices=sorted(METRIC_NAMES_BY_ENGINE),
                        default="vllm",
                        help="which server is measured: vllm (default) or llamacpp (llama-server)")
    args = parser.parse_args()

    global ENGINE, METRIC_NAMES
    ENGINE = args.engine
    METRIC_NAMES = METRIC_NAMES_BY_ENGINE[ENGINE]

    return run_measurement(args)


if __name__ == "__main__":
    sys.exit(main())
