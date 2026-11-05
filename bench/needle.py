#!/usr/bin/env python3
"""Needle-in-a-haystack check: does quantization spoil recall in long context?

Why a separate check. Perplexity is averaged over all positions and is dominated
by local n-grams, so it hardly shows whether the model can retrieve ONE fact from
the middle of a long context. That ability depends on precision: activation error
changes the keys and values, and the KV cache is 4-bit too, so the errors add up.

Each run inserts four facts, one per position, into filler text and asks four
questions. The facts are invented, so the model cannot know them from training,
and each answer is a single number.

    python3 needle.py --url http://127.0.0.1:8100/v1 --model NAME --tag base \\
        --ctx 28000 --haystack release/docs/notes/vllm-gfx1030.md --dir results/needle
    python3 needle.py --compare base a8 --dir results/needle
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

# The fact texts, questions and labels are Ukrainian on purpose: they are prompt
# data and part of the measurement.
FACTS = [
    ("шифр складання", "Шифр складання цього стенда — 74619."),
    ("номер полиці", "Запасний блок живлення лежить на полиці номер 38."),
    ("код доступу", "Код доступу до холодного сховища — 5207."),
    ("рік випуску", "Перший зразок цієї плати випущено 1997 року."),
]
QUESTIONS = [
    "Який шифр складання цього стенда? Відповідай лише числом.",
    "На якій полиці лежить запасний блок живлення? Відповідай лише числом.",
    "Який код доступу до холодного сховища? Відповідай лише числом.",
    "Якого року випущено перший зразок цієї плати? Відповідай лише числом.",
]
ANSWERS = ["74619", "38", "5207", "1997"]
POSITIONS = [0.1, 0.35, 0.6, 0.85]     # where in the filler each fact is inserted


def load_filler(path, size):
    """Repeat the text of PATH up to SIZE characters, skipping indented (code) lines."""
    text = Path(path).read_text(encoding="utf-8")
    text = "\n".join(line for line in text.split("\n") if not line.startswith("    "))
    return (text * (size // len(text) + 1))[:size]


def ask(url, model, prompt):
    body = {"model": model, "prompt": prompt, "max_tokens": 24, "temperature": 0.0}
    req = urllib.request.Request(url + "/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.load(resp)
    return data["choices"][0]["text"], data["usage"]["prompt_tokens"]


def compare(out_dir, tag1, tag2):
    d1 = json.loads((out_dir / f"needle-{tag1}.json").read_text(encoding="utf-8"))
    d2 = json.loads((out_dir / f"needle-{tag2}.json").read_text(encoding="utf-8"))
    print(f"{'position':>9} {'tokens':>8} {tag1:>10} {tag2:>10}")
    for key in d1:
        r1, r2 = d1[key], d2.get(key, {})
        print(f"{key:>9} {r1['prompt_tokens']:>8} "
              f"{('yes' if r1['ok'] else 'NO'):>10} "
              f"{('yes' if r2.get('ok') else 'NO'):>10}")
    n1 = sum(1 for k in d1 if d1[k]["ok"])
    n2 = sum(1 for k in d2 if d2[k].get("ok"))
    print(f"\n  hits: {tag1} {n1}/{len(d1)}, {tag2} {n2}/{len(d2)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="API base URL, for example http://127.0.0.1:8100/v1")
    parser.add_argument("--model", help="served model name")
    parser.add_argument("--tag", help="result file is needle-TAG.json")
    parser.add_argument("--ctx", type=int, default=28000, help="characters of filler")
    parser.add_argument("--haystack", help="text file used as the filler")
    parser.add_argument("--dir", default=".", help="directory for needle-TAG.json files")
    parser.add_argument("--compare", "--porivnjaty", nargs=2, metavar=("TAG1", "TAG2"),
                        help="print a side-by-side table of two saved runs")
    args = parser.parse_args()
    out_dir = Path(args.dir)

    if args.compare:
        compare(out_dir, *args.compare)
        return
    if not (args.url and args.model and args.tag and args.haystack):
        parser.error("--url, --model, --tag and --haystack are required unless --compare is given")

    filler = load_filler(args.haystack, args.ctx)
    results = {}
    for i, (name, fact) in enumerate(FACTS):
        for share in POSITIONS:
            pos = int(len(filler) * share)
            # Snap to a line boundary so that no word is cut.
            pos = filler.find("\n", pos) + 1 or pos
            full = (filler[:pos] + "\n\n" + fact + "\n\n" + filler[pos:]
                    + "\n\nПитання: " + QUESTIONS[i] + "\nВідповідь:")
            try:
                text, ntok = ask(args.url, args.model, full)
            except Exception as exc:
                sys.exit(f"request ({name}, {share}) failed: {exc}")
            found = ANSWERS[i] in re.sub(r"[^\d]", " ", text)
            key = f"{name[:6]}@{int(share * 100)}"
            results[key] = {"ok": bool(found), "text": text.strip()[:40], "prompt_tokens": ntok}
            print(f"  {key:>10}: {ntok:>6} tok  {'HIT' if found else 'MISS'}  {text.strip()[:30]!r}")
    n = sum(1 for k in results if results[k]["ok"])
    print(f"  total: {n}/{len(results)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"needle-{args.tag}.json").write_text(
        json.dumps(results, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
