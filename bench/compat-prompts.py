#!/usr/bin/env python3
"""Compatibility prompts for the skinny int4 kernel path.

Sends one base prompt and prompts of 9..64 tokens (counted by usage.prompt_tokens).
At prefill M <= 64 the skinny int4 kernel is used. Each request is greedy and asks
for 16 output tokens. Exit status 1 means any error, empty response or short output.

    python3 compat-prompts.py http://127.0.0.1:8100

The model name is read from MODEL_NAME (default "q9"). The prompts are written in
Ukrainian on purpose: the token counts they produce are part of the measurement.
"""
import json
import os
import sys
import threading
import urllib.request

BASE = sys.argv[1]
PROMPT = ("Поясни одним абзацом, чому тензорний паралелізм на двох "
          "відеокартах потребує швидкого обміну між ними.")
WORD = " слово"


def completion(prompt, max_tokens=16):
    body = {"model": os.environ.get("MODEL_NAME", "q9"), "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0, "ignore_eos": True}
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    return d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"]


# Prompts of 1..3 tokens. Without speculation only these give a batch with max q_len
# 2..3, which exercises the HIP attention path at VLLM_ROCM_INT4_DECODE_QLEN_MAX=3,
# both alone and mixed with decode.
TINY = ["Так", "Привіт", "Hello world", "1 2", "Ок."]


def mixed():
    """Three long decodes in the background, tiny prompts on top: mixed q_len 1 and 2..3."""
    errs = []

    def long_decode(i):
        try:
            completion(PROMPT + f" Варіант {i}.", max_tokens=300)
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=long_decode, args=(i,)) for i in range(3)]
    for t in ts:
        t.start()
    for p in TINY * 3:
        pt, ct = completion(p)
        print(f"mixed tiny prompt={pt} completion={ct} {'OK' if ct == 16 else 'SHORT'}", flush=True)
    for t in ts:
        t.join(timeout=600)
    if errs:
        raise errs[0]


def main():
    bad = 0
    cases = [PROMPT] + TINY + ["Рахуй:" + WORD * n for n in range(4, 64, 3)]
    for p in cases:
        try:
            pt, ct = completion(p)
            ok = ct == 16
            print(f"prompt={pt:3d} completion={ct} {'OK' if ok else 'SHORT'}", flush=True)
            bad += not ok
        except Exception as e:  # server failed: asking further is pointless
            print(f"ERROR on prompt of {len(p)} chars: {e!r}", flush=True)
            return 1
    try:
        mixed()
    except Exception as e:  # noqa: BLE001
        print(f"ERROR in the mixed run: {e!r}", flush=True)
        return 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
