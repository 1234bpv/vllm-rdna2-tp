#!/usr/bin/env python3
"""End-to-end check of a running vLLM server with one streamed chat request.

Usage: smoke-test.py [--url URL] [--model NAME] [--timeout S] [--needle TOKENS] [--json]

Steps:
  1. GET /v1/models. The model name comes from data[0].id unless --model is given.
  2. POST /v1/chat/completions with stream, include_usage, temperature 0 and
     max_tokens 64, with thinking switched off through chat_template_kwargs: otherwise
     a Qwen model spends all 64 tokens on reasoning and the answer stays empty. The
     answer must be non-empty, usage must be reported, and
     completion_tokens must be above zero. With --needle TOKENS the prompt is about
     TOKENS tokens of random text with a random number at its start; the answer must
     contain that number.

Standard library only. Every request has a time limit (--timeout seconds).
Exit: 0 all checks passed; 1 a check failed, or a request to the chat endpoint
failed; 2 bad call; 3 the server could not be reached at step 1.
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

PLAIN_PROMPT = "Explain in two sentences why the sky is blue."
QUESTION = ("What was the access code mentioned at the beginning? "
            "Answer with the number only.")
WORDS = ["river", "lantern", "harbor", "meadow", "compass", "orchard", "signal", "winter",
         "copper", "village", "canvas", "pattern", "garden", "window", "mountain", "ledger"]
# Rough English average, used only to size the filler text; the usage count is exact.
TOKENS_PER_WORD = 1.3
MAX_TOKENS = 64
PROMPT_SHOW = 300


def get_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_prompt(needle_tokens, rng):
    """Returns (prompt, needle). needle is None for the plain prompt."""
    if needle_tokens is None:
        return PLAIN_PROMPT, None
    code = str(rng.randint(100000, 999999))
    words = max(1, int(needle_tokens / TOKENS_PER_WORD))
    filler = " ".join(rng.choice(WORDS) for _ in range(words))
    prompt = (f"The access code is {code}. Remember it. Unrelated text follows: {filler}.\n\n"
              f"{QUESTION}")
    return prompt, code


def chat(url, model, prompt, timeout):
    """Streams one chat completion. Returns the collected text and the timing."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    text = []
    reasoning_chars = 0
    usage = None
    t_send = time.monotonic()
    t_first = None
    t_last = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if piece or reasoning:
                    now = time.monotonic()
                    if t_first is None:
                        t_first = now
                    t_last = now
                text.append(piece)
                reasoning_chars += len(reasoning)
    ttft = (t_first - t_send) if t_first is not None else None
    decode_tps = None
    if usage and usage.get("completion_tokens", 0) > 1 and t_first is not None and t_last > t_first:
        decode_tps = (usage["completion_tokens"] - 1) / (t_last - t_first)
    return {
        "text": "".join(text),
        "reasoning_chars": reasoning_chars,
        "usage": usage,
        "ttft_s": ttft,
        "decode_tok_s": decode_tps,
    }


def error_body(exc):
    try:
        return exc.read().decode("utf-8", errors="replace")[:500]
    except OSError:
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end check of a running vLLM server with one streamed chat request.",
        epilog="Exit codes: 0 passed; 1 a check failed or the chat request failed; "
               "2 bad call; 3 the server could not be reached at step 1.",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8100",
                        help="server address (default http://127.0.0.1:8100)")
    parser.add_argument("--model", help="model name (default: data[0].id from /v1/models)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="seconds per request (default 600)")
    parser.add_argument("--needle", type=int, metavar="TOKENS",
                        help="long-context mode: about TOKENS tokens with a number at the start")
    parser.add_argument("--json", action="store_true", help="print one JSON object instead of text")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be a positive number of seconds")
    if args.needle is not None and args.needle <= 0:
        parser.error("--needle must be a positive number of tokens")
    if not args.url.startswith(("http://", "https://")):
        parser.error("--url must start with http:// or https://")

    base = args.url.rstrip("/")
    say = (lambda *_: None) if args.json else print

    # Step 1: the server must answer /v1/models.
    try:
        models = get_json(base + "/v1/models", args.timeout)
    except urllib.error.HTTPError as exc:
        print(f"ERROR: GET /v1/models returned HTTP {exc.code}: {error_body(exc)}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError) as exc:
        print(f"ERROR: server not reachable at {base}: {exc}", file=sys.stderr)
        return 3
    data = models.get("data") or []
    if not data:
        print("ERROR: /v1/models lists no model", file=sys.stderr)
        return 1
    model = args.model or data[0]["id"]
    say(f"server: {base}")
    say(f"model:  {model}")

    rng = random.Random()
    prompt, needle = build_prompt(args.needle, rng)
    shown = prompt if len(prompt) <= PROMPT_SHOW else prompt[:PROMPT_SHOW] + f"... ({len(prompt)} characters)"
    say(f"prompt: {shown}")

    # Step 2: one streamed chat request.
    try:
        result = chat(base + "/v1/chat/completions", model, prompt, args.timeout)
    except urllib.error.HTTPError as exc:
        print(f"ERROR: POST /v1/chat/completions returned HTTP {exc.code}: {error_body(exc)}",
              file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"ERROR: chat request failed: {exc}", file=sys.stderr)
        return 1

    text = result["text"]
    usage = result["usage"] or {}
    completion = usage.get("completion_tokens", 0)
    checks = []

    def check(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    check("answer is not empty", text.strip(),
          f"{len(text.strip())} characters" if text.strip()
          else f"empty (reasoning text: {result['reasoning_chars']} characters)")
    check("usage is reported", result["usage"] is not None,
          f"prompt {usage.get('prompt_tokens', '?')} tokens, completion {completion} tokens")
    check("completion_tokens > 0", completion > 0, f"{completion}")
    if needle is not None:
        found = needle in text
        check("answer contains the number from the start", found,
              f"expected {needle}, answer {'contains it' if found else 'does not contain it'}")

    passed = all(c["ok"] for c in checks)
    ttft = result["ttft_s"]
    tps = result["decode_tok_s"]

    if args.json:
        print(json.dumps({
            "url": base, "model": model, "passed": passed, "checks": checks,
            "prompt": prompt, "text": text, "usage": result["usage"],
            "ttft_s": ttft, "decode_tok_s": tps,
        }, ensure_ascii=False, indent=2))
    else:
        print(f"output: {text.strip() or '(empty)'}")
        print(f"prompt tokens: {usage.get('prompt_tokens', '?')}, completion tokens: {completion}")
        print(f"time to first token: {ttft:.2f} s" if ttft is not None else "time to first token: n/a")
        print(f"decode speed: {tps:.1f} tokens/s" if tps is not None else "decode speed: n/a")
        for c in checks:
            print(f"  {'PASS' if c['ok'] else 'FAIL'} {c['name']}: {c['detail']}")
        print("RESULT:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
