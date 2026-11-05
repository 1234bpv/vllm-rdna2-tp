#!/usr/bin/env python3
"""Adds the MTP (multi-token prediction) head to a local Qwen3.5-9B AWQ directory.

The AWQ weights published as atalsharma/Qwen3.5-9B-AWQ do not contain the MTP head,
so speculative decoding has nothing to load. This script takes the 15 mtp.* tensors
from the original Qwen/Qwen3.5-9B shards. It reads only the safetensors headers and
the needed byte ranges over HTTP (the whole repository is about 19 GB, the head a few
hundred MB), writes MODEL_DIR/model-mtp.safetensors, and adds the eight MTP modules to
quantization_config.ignore so that vLLM does not try to quantize them.

Standard library only. Running it again is safe: a finished directory is left unchanged.
"""

import argparse
import json
import os
import shutil
import struct
import sys
import time
import urllib.error
import urllib.request

DEFAULT_REPO = "Qwen/Qwen3.5-9B"
# The MTP tensors sit in the last three of the four shards of the original repository.
SHARDS = [
    "model.safetensors-00002-of-00004.safetensors",
    "model.safetensors-00003-of-00004.safetensors",
    "model.safetensors-00004-of-00004.safetensors",
]
EXPECTED_TENSORS = 15
OUTPUT_NAME = "model-mtp.safetensors"
BACKUP_NAME = "config.json.before-mtp"
IGNORE_ADD = [
    "mtp.fc",
    "mtp.layers.0.mlp.down_proj",
    "mtp.layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.o_proj",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.v_proj",
]
# A safetensors header larger than this is not a real header; reject it before reading.
MAX_HEADER = 100 * 2**20


def shard_url(repo, shard):
    return f"https://huggingface.co/{repo}/resolve/main/{shard}"


def fetch(url, timeout, start=None, end=None, attempts=3):
    """One GET, or the byte range [start, end] when given. Network errors and 5xx
    answers are retried; a 4xx answer is final. A short read is an error, not silence."""
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url)
            want = None
            if start is not None:
                req.add_header("Range", f"bytes={start}-{end}")
                want = end - start + 1
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if want is not None and len(data) != want:
                raise OSError(f"got {len(data)} bytes, expected {want}")
            return data
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or attempt == attempts:
                raise
            reason = exc
        except (urllib.error.URLError, OSError) as exc:
            if attempt == attempts:
                raise
            reason = exc
        print(f"    attempt {attempt} failed ({reason}), retrying", flush=True)
        time.sleep(3)


def read_header(url, timeout):
    """Returns the safetensors header and the byte offset where the tensor data starts."""
    (length,) = struct.unpack("<Q", fetch(url, timeout, start=0, end=7))
    if length <= 0 or length > MAX_HEADER:
        raise ValueError(f"implausible safetensors header length {length}")
    header = json.loads(fetch(url, timeout, start=8, end=8 + length - 1))
    return header, 8 + length


def collect_head(repo, timeout):
    """Downloads the mtp.* tensors: name -> (dtype, shape, raw bytes)."""
    tensors = {}
    for shard in SHARDS:
        url = shard_url(repo, shard)
        print(f"reading the header of {shard}", flush=True)
        header, data_start = read_header(url, timeout)
        mine = {k: v for k, v in header.items() if k.startswith("mtp.")}
        print(f"  mtp tensors here: {len(mine)}", flush=True)
        for name in sorted(mine):
            info = mine[name]
            begin, end = info["data_offsets"]
            print(f"  {name:50s} {info['dtype']:6s} {str(info['shape']):20s} "
                  f"{(end - begin) / 2**20:8.2f} MiB", flush=True)
            data = fetch(url, timeout, start=data_start + begin, end=data_start + end - 1)
            tensors[name] = (info["dtype"], info["shape"], data)
    return tensors


def write_safetensors(path, tensors):
    """Writes the tensors as one safetensors file with a fresh, consistent header."""
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    order = sorted(tensors)
    for name in order:
        dtype, shape, data = tensors[name]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # The tensor data must start on an 8-byte boundary, so the header is padded with spaces.
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for name in order:
            f.write(tensors[name][2])
    return 8 + len(blob) + offset


def head_complete(path):
    """True when the file holds all 15 MTP tensors and its size matches the header."""
    try:
        with open(path, "rb") as f:
            (length,) = struct.unpack("<Q", f.read(8))
            if length <= 0 or length > MAX_HEADER:
                return False
            header = json.loads(f.read(length))
        size = os.path.getsize(path)
    except (OSError, struct.error, ValueError):
        return False
    mine = {k: v for k, v in header.items() if k.startswith("mtp.")}
    if len(mine) != EXPECTED_TENSORS:
        return False
    data_end = max(v["data_offsets"][1] for v in mine.values())
    return size == 8 + length + data_end


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Add the MTP head (15 mtp.* tensors) to a local Qwen3.5-9B AWQ directory.",
        epilog="Exit codes: 0 done or nothing to do; 1 download or check failed, nothing "
               "written; 2 bad call. The original config.json is kept as config.json.before-mtp.",
    )
    parser.add_argument("model_dir", metavar="MODEL_DIR", help="local model directory")
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help=f"original repository with the MTP head (default {DEFAULT_REPO})")
    parser.add_argument("--timeout", type=int, default=300,
                        help="seconds per HTTP request (default 300)")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be a positive number of seconds")

    if not os.path.isdir(args.model_dir):
        print(f"ERROR: not a directory: {args.model_dir}", file=sys.stderr)
        return 2
    config_path = os.path.join(args.model_dir, "config.json")
    if not os.path.isfile(config_path):
        print(f"ERROR: no config.json in {args.model_dir}", file=sys.stderr)
        return 2

    try:
        config = load_config(config_path)
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot read {config_path}: {exc}", file=sys.stderr)
        return 1
    qc = config.get("quantization_config")
    if not isinstance(qc, dict) or not isinstance(qc.get("ignore"), list):
        print("ERROR: config.json has no quantization_config.ignore list; "
              "this script cannot patch this checkpoint", file=sys.stderr)
        return 1

    out_path = os.path.join(args.model_dir, OUTPUT_NAME)
    head_ok = head_complete(out_path)
    ignore_ok = all(name in qc["ignore"] for name in IGNORE_ADD)
    if head_ok and ignore_ok:
        print("nothing to do: the MTP head and the ignore entries are already in place")
        return 0

    if not head_ok:
        try:
            tensors = collect_head(args.repo, args.timeout)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            print(f"ERROR: download from {args.repo} failed: {exc}", file=sys.stderr)
            print("nothing was written", file=sys.stderr)
            return 1
        if len(tensors) != EXPECTED_TENSORS:
            print(f"ERROR: expected {EXPECTED_TENSORS} mtp.* tensors, found {len(tensors)}; "
                  "nothing was written", file=sys.stderr)
            return 1
        tmp_path = out_path + ".tmp"
        try:
            size = write_safetensors(tmp_path, tensors)
            os.replace(tmp_path, out_path)
        except OSError as exc:
            print(f"ERROR: cannot write {out_path}: {exc}", file=sys.stderr)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return 1
        print(f"wrote {OUTPUT_NAME}: {size / 2**20:.1f} MiB, {len(tensors)} tensors")
    else:
        print(f"{OUTPUT_NAME} is already complete; updating config.json only")

    if not ignore_ok:
        backup = os.path.join(args.model_dir, BACKUP_NAME)
        if not os.path.exists(backup):
            shutil.copyfile(config_path, backup)
        added = 0
        for name in IGNORE_ADD:
            if name not in qc["ignore"]:
                qc["ignore"].append(name)
                added += 1
        tmp_cfg = config_path + ".tmp"
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_cfg, config_path)
        print(f"config.json: added {added} ignore entries; original kept as {BACKUP_NAME}")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
