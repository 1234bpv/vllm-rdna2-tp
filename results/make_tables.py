#!/usr/bin/env python3
"""Generate the performance tables of the documentation from measurement records.

Every number in the generated tables comes from kv-concurrency.jsonl, selected by
runs.json, or from all-reduce.json. Nothing is typed by hand into the documents:
the tables live between marker comments and this script rewrites them.

    <!-- table:NAME -->
    ...generated...
    <!-- /table:NAME -->

Usage:
    python3 results/make_tables.py            rewrite the tables and summary.csv
    python3 results/make_tables.py --check    exit 1 if anything is out of date
    python3 results/make_tables.py --list     print the known table names

Definitions (see docs/PERFORMANCE.md):
    PP   prompt processing, tokens/s: the sum of prompt tokens of all streams
         divided by the time until the last stream received its first token.
    TG   decode, tokens/s: the record's total decode speed in the window when
         all streams were generating (field sukupna_shvydkist_tok_s).
    TTFT time until the last stream received its first token, seconds.

    A run with the field "older" takes the points it has no records for from that older run
    (recursively). Such a point is marked " *" in the tables, so that a reader sees which numbers
    were measured on an earlier day.
"""
import copy
import csv
import io
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DOCUMENTS = [ROOT / "README.md", ROOT / "docs" / "PERFORMANCE.md"]
SUMMARY_CSV = HERE / "summary.csv"


class Point:
    """Numbers of one (prompt length, streams) point of one run."""

    def __init__(self, record):
        threads = record["chyslo"].get("potoky") or []
        prompts = [t["prompt_tokens"] for t in threads]
        ttfts = [t["chas_do_pershoho_tokena_s"] for t in threads]
        self.stage = record["etap"]
        self.when = record["chas"]
        self.ttft = max(ttfts)
        self.pp = sum(prompts) / self.ttft
        self.tg = record["chyslo"].get("sukupna_shvydkist_tok_s")
        self.preempted = record["chyslo"].get("vytisnen_riznytsia")
        self.older = False


def is_valid(record):
    """A record counts only if every stream finished without an error."""
    threads = record["chyslo"].get("potoky") or []
    if not threads:
        return False
    if (record.get("dokaz") or {}).get("pomylky"):
        return False
    return all(
        not t.get("pomylka") and t.get("prompt_tokens") and t.get("chas_do_pershoho_tokena_s")
        for t in threads
    )


def rule_matches(record, rule):
    if record["etap"].split(" ")[0] != rule["stage"]:
        return False
    if "contains" in rule and rule["contains"] not in record["etap"]:
        return False
    if "before" in rule and not record["chas"] < rule["before"]:
        return False
    return True


def merged_points(run_id, by_id, own, done, chain=()):
    """Own points of a run, plus the points of its older run that it lacks, marked as older."""
    if run_id in done:
        return done[run_id]
    if run_id in chain:
        raise SystemExit(f"run {run_id}: older runs form a cycle: {' -> '.join(chain + (run_id,))}")
    points = dict(own[run_id])
    older_id = by_id[run_id].get("older")
    if older_id is not None:
        if older_id not in by_id:
            raise SystemExit(f"run {run_id}: older run {older_id} is not in runs.json")
        for key, point in merged_points(older_id, by_id, own, done, chain + (run_id,)).items():
            if key not in points:
                point = copy.copy(point)
                point.older = True
                points[key] = point
    done[run_id] = points
    return points


def load():
    manifest = json.loads((HERE / "runs.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (HERE / "kv-concurrency.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    own = {}
    for run in manifest["runs"]:
        points = {}
        for record in records:
            if not any(rule_matches(record, rule) for rule in run["match"]):
                continue
            if not is_valid(record):
                continue
            umovy = record["umovy"]
            # The last valid record of a point wins: repeats follow fixes.
            points[(umovy["promt_tokeniv_zadano"], umovy["potokiv"])] = Point(record)
        if not points:
            raise SystemExit(f"run {run['id']}: no valid records")
        own[run["id"]] = points
    # All runs are read before any "older" is followed, so the order of the runs in the file does not matter.
    by_id = {run["id"]: run for run in manifest["runs"]}
    done = {}
    runs = {}
    for run_id, run in by_id.items():
        runs[run_id] = dict(run, points=merged_points(run_id, by_id, own, done))
    all_reduce = json.loads((HERE / "all-reduce.json").read_text(encoding="utf-8"))
    return manifest, runs, all_reduce


def fmt_pp(value):
    return "—" if value is None else f"{value:.0f}"


def fmt_tg(value):
    return "—" if value is None else f"{value:.1f}"


def fmt_ttft(value):
    if value is None:
        return "—"
    return f"{value:.1f}" if value < 10 else f"{value:.0f}"


def fmt_tokens(value):
    return "—" if value is None else f"{value:,}"


def fmt_point(length, streams):
    if length >= 1000:
        short = f"{length / 1000:.0f}k" if length % 1000 == 0 else f"{length / 1024:.0f}k"
    else:
        short = str(length)
    return f"{short} × {streams}"


def table(header, rows, align=None):
    align = align or ["---"] + ["---:"] * (len(header) - 1)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def get(run, length, streams, attr):
    point = run["points"].get((length, streams))
    return None if point is None else getattr(point, attr)


def mark(run, length, streams):
    """Suffix of a table cell whose point was taken from an older run."""
    point = run["points"].get((length, streams))
    return " *" if point is not None and point.older else ""


def marked(run, length, streams, attr, fmt):
    value = get(run, length, streams, attr)
    return fmt(value) + ("" if value is None else mark(run, length, streams))


def show(value):
    return "—" if value is None else str(value)


def cell(run, length, streams, attr, fmt):
    """Value of a point, or of the run's documented alternative prompt length."""
    value = get(run, length, streams, attr)
    if value is not None:
        return fmt(value) + mark(run, length, streams)
    alternative = run.get("alt_points", {}).get(str(length))
    if alternative is not None:
        value = get(run, alternative, streams, attr)
        if value is not None:
            return f"{fmt(value)} ({alternative // 1024}k){mark(run, alternative, streams)}"
    return "—"


def short_date(run):
    return run["date"]


SUMMARY_ROWS = ["rel-q27", "rel-q27-mtp", "rel-q9", "rel-q9-mtp", "rel-q35", "rel-q35-mtp", "rel-q36", "q36-mtp",
                "rel-q36-mtp-262k", "rel-gemma"]
# The summary rows, the release runs of the other engines and the pipeline, then the older runs that
# the release runs refer to or that are shown on their own.
CONFIG_ROWS = SUMMARY_ROWS + ["rel-llama-q36", "rel-llama-q27-3cards", "rel-pp3-q27", "pp3-gemma",
                              "q27", "q27-mtp", "q9", "q9-mtp", "q35", "q35-mtp", "q35-offload", "q36",
                              "q36-mtp-262k", "q36-mtp-262k-kv8", "gemma", "gemma-streams", "llama-q36",
                              "llama-q27-3cards", "pp3-q27"]


def t_summary(runs):
    rows = []
    for run_id in SUMMARY_ROWS:
        run = runs[run_id]
        rows.append([
            show(run.get("model")) + (" + MTP" if run.get("mtp") else ""),
            show(run.get("weights") and run["weights"].split(",")[0]),
            fmt_tokens(run.get("window")),
            fmt_tokens(run.get("kv_capacity")),
            cell(run, 300, 1, "pp", fmt_pp),
            cell(run, 32768, 1, "pp", fmt_pp),
            cell(run, 125000, 1, "pp", fmt_pp),
            cell(run, 125000, 1, "ttft", fmt_ttft),
            cell(run, 300, 1, "tg", fmt_tg),
            cell(run, 32768, 1, "tg", fmt_tg),
            cell(run, 125000, 1, "tg", fmt_tg),
            cell(run, 300, 4, "tg", fmt_tg),
            cell(run, 125000, 2, "tg", fmt_tg),
        ])
    return table(
        ["Model", "Weights", "Context", "KV cache, tokens", "PP 300", "PP 32k", "PP 125k", "TTFT 125k, s",
         "TG 300", "TG 32k", "TG 125k", "ΣTG 300 × 4", "ΣTG 125k × 2"],
        rows,
        ["---", "---"] + ["---:"] * 11,
    )


def t_configs(runs):
    rows = []
    for run_id in CONFIG_ROWS:
        run = runs[run_id]
        rows.append([
            f"`{run_id}`",
            show(run.get("model")),
            show(run.get("checkpoint")),
            show(run.get("weights")),
            show(run.get("engine")),
            show(run.get("cards")),
            fmt_tokens(run.get("window")),
            show(run.get("kv")),
            show(run.get("util")),
            show(run.get("max_seqs")),
            show(run.get("tree")),
            show(run.get("date")),
        ])
    return table(
        ["Run", "Model", "Checkpoint", "Weights", "Engine", "Cards", "Context", "KV", "UTIL", "Max streams", "Code", "Date"],
        rows,
        ["---", "---", "---", "---", "---", "---:", "---:", "---", "---:", "---:", "---", "---"],
    )


def t_flags(runs):
    rows = []
    for run_id in CONFIG_ROWS:
        rows.append([f"`{run_id}`", show(runs[run_id].get("flags"))])
    return table(["Run", "Settings beyond the common command"], rows, ["---", "---"])


def matrix(runs, run_ids, labels):
    """Rows are points, columns are runs; each cell is PP / TG."""
    keys = sorted({key for run_id in run_ids for key in runs[run_id]["points"]})
    rows = []
    for key in keys:
        row = [fmt_point(*key)]
        for run_id in run_ids:
            pp = get(runs[run_id], key[0], key[1], "pp")
            tg = get(runs[run_id], key[0], key[1], "tg")
            if pp is None:
                row.append("—")
            elif key[0] < 1000 and key[1] > 1:
                # Total PP of short prompts measures warm-up, not throughput;
                # repeats of the same point differ by up to 75 %.
                row.append(f"— / {fmt_tg(tg)}{mark(runs[run_id], key[0], key[1])}")
            else:
                row.append(f"{fmt_pp(pp)} / {fmt_tg(tg)}{mark(runs[run_id], key[0], key[1])}")
        rows.append(row)
    return table(["Prompt × streams"] + labels, rows)


def t_matrix_q27(runs):
    return matrix(runs, ["rel-q27", "rel-q27-mtp"], ["No MTP, PP / TG", "MTP 2, PP / TG"])


def t_matrix_q9(runs):
    return matrix(runs, ["rel-q9", "rel-q9-mtp"], ["No MTP, PP / TG", "MTP 2, PP / TG"])


def t_matrix_q35(runs):
    return matrix(runs, ["rel-q35", "q35-offload", "rel-q35-mtp"],
                  ["No MTP, PP / TG", "No MTP, KV offload 32 GiB, 1 Oct, PP / TG", "MTP 2, PP / TG"])


def t_matrix_q36(runs):
    return matrix(runs, ["rel-q36", "q36-mtp", "rel-q36-mtp-262k", "q36-mtp-262k-kv8"],
                  ["No MTP, 131k, PP / TG", "MTP 2, 131k, 7 Oct, PP / TG", "MTP 2, 262k, PP / TG",
                   "MTP 2, 262k, int8 KV, 7 Oct, PP / TG"])


def t_matrix_gemma(runs):
    return matrix(runs, ["rel-gemma", "gemma-streams"],
                  ["HIP prefill kernel, PP / TG", "Before the HIP prefill kernel, PP / TG"])


def ratio(a, b):
    if a is None or b in (None, 0):
        return "—"
    return f"×{a / b:.2f}"


def comparison(runs, fork_id, other_id, points):
    fork, other = runs[fork_id], runs[other_id]
    rows = []
    for length, streams in points:
        # A 300-token prompt is dominated by warm-up, so its PP is not compared.
        short = length < 1000
        rows.append([
            fmt_point(length, streams),
            "—" if short else marked(fork, length, streams, "pp", fmt_pp),
            marked(fork, length, streams, "tg", fmt_tg),
            marked(fork, length, streams, "ttft", fmt_ttft),
            "—" if short else marked(other, length, streams, "pp", fmt_pp),
            marked(other, length, streams, "tg", fmt_tg),
            marked(other, length, streams, "ttft", fmt_ttft),
            "—" if short else ratio(get(fork, length, streams, "pp"), get(other, length, streams, "pp")),
            ratio(get(fork, length, streams, "tg"), get(other, length, streams, "tg")),
        ])
    return rows


def t_vs_llamacpp(runs):
    header = ["Prompt × streams", "Fork PP", "Fork TG", "Fork TTFT, s",
              "llama.cpp PP", "llama.cpp TG", "llama.cpp TTFT, s", "PP ratio", "TG ratio"]
    points = [(300, 1), (300, 4), (32768, 1), (32768, 2), (125000, 1), (125000, 2)]
    return table(header, comparison(runs, "rel-q36", "rel-llama-q36", points))


def t_vs_llamacpp_q27(runs):
    header = ["Prompt × streams", "Fork PP", "Fork TG", "Fork TTFT, s",
              "llama.cpp PP", "llama.cpp TG", "llama.cpp TTFT, s", "PP ratio", "TG ratio"]
    points = [(300, 1), (32768, 1), (125000, 1), (125000, 2)]
    return table(header, comparison(runs, "rel-q27", "rel-llama-q27-3cards", points))


def t_vs_pp3(runs):
    header = ["Model", "Prompt × streams", "TP=2, 2 cards: PP", "TP=2, 2 cards: TG",
              "Pipeline, 3 cards: PP", "Pipeline, 3 cards: TG", "PP ratio", "TG ratio"]
    rows = []
    for model, fork_id, pp3_id, points in [
        ("Qwen3.8-27B", "rel-q27", "rel-pp3-q27", [(300, 1), (32768, 1), (125000, 1), (125000, 2)]),
        ("Gemma 4 26B-A4B, 3–4 Oct", "gemma", "pp3-gemma", [(250000, 1)]),
        ("Gemma 4 26B-A4B, 3–4 Oct", "gemma-streams", "pp3-gemma", [(125000, 4), (64000, 8)]),
    ]:
        for length, streams in points:
            fork, pp3 = runs[fork_id], runs[pp3_id]
            rows.append([
                model,
                fmt_point(length, streams),
                marked(fork, length, streams, "pp", fmt_pp),
                marked(fork, length, streams, "tg", fmt_tg),
                marked(pp3, length, streams, "pp", fmt_pp),
                marked(pp3, length, streams, "tg", fmt_tg),
                ratio(get(fork, length, streams, "pp"), get(pp3, length, streams, "pp")),
                ratio(get(fork, length, streams, "tg"), get(pp3, length, streams, "tg")),
            ])
    return table(header, rows, ["---", "---"] + ["---:"] * 6)


def t_tp_scaling(runs):
    one, two = runs["q9-tp1"], runs["q9-tp2-0919"]
    rows = []
    for length, streams in [(125000, 1), (125000, 4)]:
        rows.append([
            fmt_point(length, streams),
            fmt_pp(get(one, length, streams, "pp")),
            fmt_tg(get(one, length, streams, "tg")),
            fmt_pp(get(two, length, streams, "pp")),
            fmt_tg(get(two, length, streams, "tg")),
            ratio(get(two, length, streams, "pp"), get(one, length, streams, "pp")),
            ratio(get(two, length, streams, "tg"), get(one, length, streams, "tg")),
        ])
    return table(["Prompt × streams", "1 card PP", "1 card TG", "2 cards PP", "2 cards TG", "PP ratio", "TG ratio"], rows)


def t_kv(runs):
    kv4, kv8 = runs["q36-mtp-262k"], runs["q36-mtp-262k-kv8"]
    rows = [["KV cache, tokens", fmt_tokens(kv4["kv_capacity"]), fmt_tokens(kv8["kv_capacity"]),
             ratio(kv4["kv_capacity"], kv8["kv_capacity"])]]
    for length, streams in [(300, 1), (32768, 1), (32768, 2), (125000, 1), (125000, 2)]:
        rows.append([
            f"TG {fmt_point(length, streams)}",
            fmt_tg(get(kv4, length, streams, "tg")),
            fmt_tg(get(kv8, length, streams, "tg")),
            ratio(get(kv4, length, streams, "tg"), get(kv8, length, streams, "tg")),
        ])
    for length, streams in [(32768, 1), (125000, 1)]:
        rows.append([
            f"PP {fmt_point(length, streams)}",
            fmt_pp(get(kv4, length, streams, "pp")),
            fmt_pp(get(kv8, length, streams, "pp")),
            ratio(get(kv4, length, streams, "pp"), get(kv8, length, streams, "pp")),
        ])
    return table(["Qwen3.6-35B-A3B, MTP 2, context 262,144", "int4 KV", "int8 KV", "int4 / int8"], rows)


COMPONENTS = [
    # title, baseline run, improved run, metric, points
    ("HIP decode attention kernel, Qwen3.8-27B", "q27-attn-triton", "q27-attn-hip", "tg", [(125000, 1), (125000, 2), (32768, 4)]),
    ("HIP prefill kernel, Gemma 4 26B-A4B", "gemma-prefill-triton", "gemma", "pp", [(32768, 1), (125000, 1), (250000, 1)]),
    ("HIP prefill kernel, Qwen3.6-35B-A3B", "q36-prefill-triton", "q36-prefill-hip", "pp", [(32768, 1)]),
    ("W8A16 kernel, Qwen3.6-35B-A3B", "q36-w8-off-1", "q36-w8-on-1", "tg", [(32768, 1), (32768, 4), (300, 4)]),
    ("W8A16 kernel, repeat", "q36-w8-off-2", "q36-w8-on-2", "tg", [(32768, 1), (32768, 4), (300, 4)]),
]


def t_components(runs):
    rows = []
    for title, base_id, new_id, attr, points in COMPONENTS:
        fmt = fmt_pp if attr == "pp" else fmt_tg
        for length, streams in points:
            base = get(runs[base_id], length, streams, attr)
            new = get(runs[new_id], length, streams, attr)
            rows.append([title, attr.upper(), fmt_point(length, streams), marked(runs[base_id], length, streams, attr, fmt),
                         marked(runs[new_id], length, streams, attr, fmt), ratio(new, base)])
    return table(["Component", "Metric", "Prompt × streams", "Without", "With", "Ratio"], rows,
                 ["---", "---", "---", "---:", "---:", "---:"])


REPEATS = [
    ("Gemma 4 26B-A4B, same tree", "gemma-rep1", "gemma-rep2", [(23000, 1), (100000, 1)]),
    ("Qwen3.6-35B-A3B, W8A16 on, same tree", "q36-w8-on-1", "q36-w8-on-2", [(32768, 1), (32768, 4), (300, 4)]),
    ("Qwen3.6-35B-A3B, W8A16 off, same tree", "q36-w8-off-1", "q36-w8-off-2", [(32768, 1), (32768, 4), (300, 4)]),
    ("Qwen3.8-27B, neighbouring commits", "q27-rep-a", "q27", [(300, 1), (300, 4)]),
    ("Qwen3.8-27B, same tree", "q27-rep-b", "q27", [(300, 1), (300, 4)]),
    ("Qwen3.8-27B, 1 Oct and release tree", "q27", "rel-q27", [(300, 1), (32768, 1), (125000, 1), (125000, 2)]),
    ("Qwen3.6-35B-A3B, 7 Oct and release tree", "q36", "rel-q36", [(32768, 1), (125000, 1), (125000, 2)]),
    ("Qwen3.8-27B, release tree, two starts", "ab-q27-1-on", "ab-q27-2-on", [(300, 1), (32768, 1)]),
    ("Qwen3.6-35B-A3B + MTP, release tree, two starts", "rel-q36-mtp-262k", "rep-q36-mtp-262k-2",
     [(300, 1), (32768, 1), (125000, 1), (125000, 2)]),
]


def spread(a, b):
    if a is None or b is None:
        return "—"
    return f"{abs(a - b) / ((a + b) / 2) * 100:.1f} %"


def t_repeats(runs):
    rows = []
    for title, a_id, b_id, points in REPEATS:
        for length, streams in points:
            a, b = runs[a_id], runs[b_id]
            tg_a, tg_b = get(a, length, streams, "tg"), get(b, length, streams, "tg")
            pp_a, pp_b = get(a, length, streams, "pp"), get(b, length, streams, "pp")
            rows.append([
                title,
                fmt_point(length, streams),
                f"{marked(a, length, streams, 'tg', fmt_tg)} / {marked(b, length, streams, 'tg', fmt_tg)}",
                spread(tg_a, tg_b),
                f"{marked(a, length, streams, 'pp', fmt_pp)} / {marked(b, length, streams, 'pp', fmt_pp)}",
                spread(pp_a, pp_b),
            ])
    return table(["Pair of runs", "Prompt × streams", "TG, run A / run B", "TG spread", "PP, run A / run B", "PP spread"],
                 rows, ["---", "---", "---:", "---:", "---:", "---:"])


def t_all_reduce(all_reduce):
    rows = []
    for row in all_reduce["latency"]:
        rows.append([
            f"{row['elements']:,}" + (" (decode exchange)" if row.get("decode") else ""),
            f"{row['bytes'] / 1024:.0f} KiB",
            f"{row['fork_us']:.1f}",
            f"~{row['rccl_us']:.0f}",
            f"×{row['rccl_us'] / row['fork_us']:.1f}",
        ])
    return table(["fp16 elements", "Size", "This fork, µs", "RCCL, µs", "RCCL / fork"], rows)


def t_all_reduce_cost(all_reduce):
    cost = all_reduce["all_collectives"]
    rows = [
        ["With all exchanges", f"{cost['with_exchange_tok_s']:.2f}", f"{cost['with_exchange_step_ms']:.1f}"],
        ["No exchange at all (wrong output, timing only)", f"{cost['without_exchange_tok_s']:.2f}",
         f"{cost['without_exchange_step_ms']:.1f}"],
        ["Cost of all exchanges", "—",
         f"{cost['with_exchange_step_ms'] - cost['without_exchange_step_ms']:.1f} "
         f"({(cost['with_exchange_step_ms'] - cost['without_exchange_step_ms']) / cost['with_exchange_step_ms'] * 100:.0f} %)"],
    ]
    return table(["Decode of Qwen3.8-27B, one stream", "TG, tokens/s", "Step, ms"], rows)


def t_decode_chain(all_reduce):
    rows = []
    previous = None
    for step in all_reduce["decode_chain"]:
        rows.append([step["change"], f"{step['tok_s']:.2f}",
                     "—" if previous is None else f"×{step['tok_s'] / previous:.2f}"])
        previous = step["tok_s"]
    return table(["Change, applied in this order", "TG, tokens/s", "Step gain"], rows)


# A/B runs of the host all-reduce: (model, runs with this fork's all-reduce on, runs with RCCL, points).
# A row averages the repeats of each side.
ALL_REDUCE_AB = [
    ("Qwen3.8-27B", ["ab-q27-1-on", "ab-q27-2-on", "ab-q27-3-on"],
     ["ab-q27-1-off", "ab-q27-2-off", "ab-q27-3-off"], [(300, 1), (32768, 1)]),
    ("Qwen3.5-9B", ["ab-q9-on"], ["ab-q9-off"], [(300, 1), (300, 4), (8192, 1), (8192, 4)]),
    ("Qwen3.6-35B-A3B", ["ab-q36-on"], ["ab-q36-off"], [(300, 1), (300, 4), (32768, 1), (32768, 2)]),
]


def mean_of(runs, run_ids, length, streams, attr):
    """Mean over the repeats. Every repeat must have the value, or the mean would hide a gap."""
    values = []
    for run_id in run_ids:
        point = runs[run_id]["points"].get((length, streams))
        value = None if point is None else getattr(point, attr)
        if value is None:
            raise SystemExit(f"all-reduce-ab: run {run_id} has no {attr} at {fmt_point(length, streams)}")
        values.append(value)
    return sum(values) / len(values)


def t_all_reduce_ab(runs):
    rows = []
    for model, on_ids, off_ids, points in ALL_REDUCE_AB:
        for length, streams in points:
            tg_on = mean_of(runs, on_ids, length, streams, "tg")
            tg_off = mean_of(runs, off_ids, length, streams, "tg")
            # PP of short prompts measures warm-up, not throughput: not shown (see comparison()).
            short = length < 1000
            pp_on = None if short else mean_of(runs, on_ids, length, streams, "pp")
            pp_off = None if short else mean_of(runs, off_ids, length, streams, "pp")
            rows.append([
                model,
                fmt_point(length, streams),
                fmt_tg(tg_off),
                fmt_tg(tg_on),
                f"{(tg_on / tg_off - 1) * 100:+.1f} %",
                fmt_pp(pp_off),
                fmt_pp(pp_on),
                f"{len(on_ids)} / {len(off_ids)}",
            ])
    return table(["Model", "Prompt × streams", "TG, RCCL", "TG, this fork", "TG gain", "PP, RCCL", "PP, this fork", "Runs"],
                 rows, ["---", "---"] + ["---:"] * 6)


TABLES = {
    "summary": lambda runs, ar: t_summary(runs),
    "configs": lambda runs, ar: t_configs(runs),
    "flags": lambda runs, ar: t_flags(runs),
    "matrix-q27": lambda runs, ar: t_matrix_q27(runs),
    "matrix-q9": lambda runs, ar: t_matrix_q9(runs),
    "matrix-q35": lambda runs, ar: t_matrix_q35(runs),
    "matrix-q36": lambda runs, ar: t_matrix_q36(runs),
    "matrix-gemma": lambda runs, ar: t_matrix_gemma(runs),
    "vs-llamacpp": lambda runs, ar: t_vs_llamacpp(runs),
    "vs-llamacpp-q27": lambda runs, ar: t_vs_llamacpp_q27(runs),
    "vs-pp3": lambda runs, ar: t_vs_pp3(runs),
    "tp-scaling": lambda runs, ar: t_tp_scaling(runs),
    "kv": lambda runs, ar: t_kv(runs),
    "components": lambda runs, ar: t_components(runs),
    "repeats": lambda runs, ar: t_repeats(runs),
    "all-reduce": lambda runs, ar: t_all_reduce(ar),
    "all-reduce-ab": lambda runs, ar: t_all_reduce_ab(runs),
    "all-reduce-cost": lambda runs, ar: t_all_reduce_cost(ar),
    "decode-chain": lambda runs, ar: t_decode_chain(ar),
}

MARKER = re.compile(r"(<!-- table:([a-z0-9-]+) -->\n).*?(\n?<!-- /table:\2 -->)", re.S)


def render_document(text, generated):
    used = set()

    def replace(match):
        name = match.group(2)
        if name not in generated:
            raise SystemExit(f"unknown table in document: {name}")
        used.add(name)
        return match.group(1) + generated[name] + "\n<!-- /table:" + name + " -->"

    return MARKER.sub(replace, text), used


def render_csv(runs):
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["run", "model", "engine", "cards", "tp", "pp_mode", "mtp", "context", "kv", "code", "date",
                     "prompt_tokens", "streams", "pp_tok_s", "tg_tok_s", "ttft_s", "preempted"])
    for run_id, run in sorted(runs.items()):
        for (length, streams), point in sorted(run["points"].items()):
            if point.older:
                continue  # listed under the run that measured it
            writer.writerow([
                run_id, run["model"], run["engine"], run["cards"], run["tp"], run["pp"], int(run["mtp"]),
                run["window"], run["kv"], run["tree"], run["date"], length, streams,
                f"{point.pp:.1f}", "" if point.tg is None else f"{point.tg:.3f}", f"{point.ttft:.3f}",
                "" if point.preempted is None else point.preempted,
            ])
    return out.getvalue()


def main(argv):
    if len(argv) > 1 or (argv and argv[0] not in ("--check", "--list")):
        print(__doc__)
        return 2
    if argv == ["--list"]:
        print("\n".join(TABLES))
        return 0
    _, runs, all_reduce = load()
    generated = {name: build(runs, all_reduce) for name, build in TABLES.items()}
    stale = []
    used_anywhere = set()
    for path in DOCUMENTS:
        if not path.exists():
            continue
        # Bytes, not text mode: text mode hides CRLF, and --check promises an exact match.
        text = path.read_bytes().decode("utf-8")
        new_text, used = render_document(text, generated)
        new_text = new_text.replace("\r\n", "\n")
        used_anywhere |= used
        if new_text != text:
            stale.append(path.relative_to(ROOT).as_posix())
            if not argv:
                path.write_text(new_text, encoding="utf-8", newline="\n")
    csv_text = render_csv(runs)
    current_csv = SUMMARY_CSV.read_bytes().decode("utf-8") if SUMMARY_CSV.exists() else None
    if csv_text != current_csv:
        stale.append(SUMMARY_CSV.relative_to(ROOT).as_posix())
        if not argv:
            SUMMARY_CSV.write_text(csv_text, encoding="utf-8", newline="\n")
    unused = sorted(set(TABLES) - used_anywhere)
    if unused:
        print("tables not used in any document:", ", ".join(unused))
    if argv == ["--check"]:
        if stale:
            print("out of date:", ", ".join(stale))
            return 1
        print("all tables are up to date")
        return 0
    print("updated:", ", ".join(stale) if stale else "nothing, already up to date")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main(sys.argv[1:]))
