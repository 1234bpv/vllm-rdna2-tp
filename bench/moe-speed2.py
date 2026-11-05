#!/usr/bin/env python3
"""Aggregate decode speed of several simultaneous streams, in the window where ALL streams decode.

The speed is counted from the server's token counter, between the first token of the
last stream to start and the end of the first stream to finish. Use it for multi-stream
comparisons instead of summing per-stream speeds. Each stream's speed there is measured
in its own window, and the windows do not line up: a stream with a cached prefix decodes
on its own while the others are still in prefill, so the sum depends on how long the
other streams spent in prefill. This script avoids that:
* prompts are unique to each run (a random salt at the start), so the prefix cache does not apply;
* tokens are read from vllm:generation_tokens_total on /metrics, polled every 0.25 s;
* output is long (768 tokens by default), so the shared window lasts several seconds.

    python3 moe-speed2.py URL MODEL WORDS STREAMS [OUTPUT_TOKENS]

    URL            server base URL, for example http://127.0.0.1:8100
    MODEL          served model name
    WORDS          prompt length in filler words
    STREAMS        number of simultaneous streams
    OUTPUT_TOKENS  tokens per stream (default 768)
    SPEED2_SALT    optional fixed salt, for reproducing a run

Exit status 1 means any error, incomplete output, or a shared window shorter than 2 s.
"""
import json
import os
import random
import sys
import threading
import time
import urllib.request

URL, MODEL = sys.argv[1], sys.argv[2]
WORDS, THREADS = int(sys.argv[3]), int(sys.argv[4])
OUT = int(sys.argv[5]) if len(sys.argv) > 5 else 768
SALT = int(os.environ["SPEED2_SALT"]) if os.environ.get("SPEED2_SALT") else random.randrange(10**9)
results, errors, samples, all_ids, texts = {}, [], [], {}, {}
stop = threading.Event()


def gen_tokens():
    """Total generated tokens, from the server counter on /metrics."""
    with urllib.request.urlopen(URL + "/metrics", timeout=10) as resp:
        total = 0.0
        for line in resp.read().decode().splitlines():
            if line.startswith("vllm:generation_tokens_total"):
                total += float(line.rsplit(" ", 1)[1])
        return total


def poll():
    while not stop.is_set():
        try:
            samples.append((time.time(), gen_tokens()))
        except Exception as exc:  # noqa: BLE001
            errors.append("metrics: " + repr(exc))
            return
        time.sleep(0.25)


def run_stream(i):
    # The prompt text is Ukrainian on purpose ("Сіль" = salt, "слово" = word): it is
    # the measured input, so it stays as it is.
    prompt = (f"Сіль {SALT}-{i}. " + " ".join(f"слово{(j * 7 + i) % 97}" for j in range(WORDS))
              + "\nПерекажи коротко.")
    body = {"model": MODEL, "prompt": prompt, "max_tokens": OUT, "temperature": 0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
            "return_token_ids": True}   # token IDs identify the stream's tokens without the text
    req = urllib.request.Request(URL + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t_start = time.time(); t_first = None; usage = None; ids = []; text = []
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for line in resp:
                line = line.strip()
                if not line.startswith(b"data:") or line == b"data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if d.get("choices"):
                    ids.extend(d["choices"][0].get("token_ids") or [])
                    text.append(d["choices"][0].get("text") or "")
                # First TOKEN, not first text: a stream that degenerates into incomplete
                # UTF-8 bytes has no text at all but still generates tokens.
                if ids and t_first is None:
                    t_first = time.time()
                if d.get("usage"):
                    usage = d["usage"]
        results[i] = (usage["prompt_tokens"], usage["completion_tokens"], t_start, t_first, time.time())
        all_ids[i] = ids
        texts[i] = "".join(text)
    except Exception as exc:  # noqa: BLE001
        errors.append(repr(exc))


poller = threading.Thread(target=poll)
poller.start()
threads = [threading.Thread(target=run_stream, args=(i,)) for i in range(THREADS)]
for t in threads:
    t.start()
for t in threads:
    t.join(timeout=3600)
stop.set()
poller.join(timeout=30)
if errors or len(results) != THREADS:
    print("ERROR:", errors or f"finished {len(results)} of {THREADS}")
    sys.exit(1)
if any(r[1] != OUT for r in results.values()):
    print("ERROR: incomplete output", [r[1] for r in results.values()])
    sys.exit(1)
empty = [i for i in sorted(results) if not texts.get(i)]
if empty:   # not a measurement failure: tokens were generated, but the model degenerated
    from collections import Counter
    print(f"WARNING (salt {SALT}): streams without text {empty}, degenerate continuation")
    for i in empty:
        print(f"    stream {i}: most common ids {Counter(all_ids[i]).most_common(3)}")
if any(r[3] is None for r in results.values()):
    print("ERROR: streams without any token (does the server return token_ids?)")
    sys.exit(1)
window_start = max(r[3] for r in results.values())   # all streams are decoding now
window_end = min(r[4] for r in results.values())     # the first stream has finished
inside = [(t, g) for t, g in samples if window_start <= t <= window_end]
if window_end - window_start < 2 or len(inside) < 4:
    print(f"ERROR: shared window {window_end - window_start:.2f} s, samples {len(inside)}")
    sys.exit(1)
(ta, ga), (tb, gb) = inside[0], inside[-1]
for i in sorted(results):
    p, c, t0, t1, t2 = results[i]
    print(f"stream {i}: prompt={p} TTFT {t1 - t0:.2f} s, own decode {(c - 1) / (t2 - t1):.2f} tok/s")
print(f"SHARED window {tb - ta:.2f} s: {gb - ga:.0f} server tokens, "
      f"total {(gb - ga) / (tb - ta):.2f} tok/s on {THREADS} streams")
