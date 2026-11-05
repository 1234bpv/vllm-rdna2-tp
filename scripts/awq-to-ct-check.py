#!/usr/bin/env python3
"""Checks the output of awq-to-ct.py. Run with $VLLM_RDNA2_HOME/venv/bin/python;
no GPU is needed.

1. Numbers: for selected quantized modules, the input is dequantized with the AWQ
   semantics (an independent, FORWARD formula: slot s of an int32 goes to out-index
   AWQ_ORDER[s]; the converter does the inverse permutation) and compared with
   torch.equal against the output dequantized by the compressed_tensors library
   decompressor (PackedQuantizationCompressor.decompress).
   Without --dst the output is the result of convert_module in memory. With --dst,
   if a converted checkpoint is already there, its tensors in the files are checked too.
2. Names: the set of output tensors equals the one expected from the input index
   (each quantized module gives exactly 4 fields, the rest is copied, nothing extra);
   non-quantized tensors are compared bit for bit on a sample.
3. Config: quantization_config passes QuantizationConfig from compressed_tensors, and
   the output ignore list equals the input modules_to_not_convert.
"""

import argparse
import importlib.util
import json
import os
import re
import sys

import torch
from compressed_tensors.compressors.pack_quantized import PackedQuantizationCompressor
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
)
from safetensors import safe_open

AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
FIELDS = ("weight_packed", "weight_scale", "weight_shape", "weight_zero_point")

# The converter is loaded from this directory, so the check always tests the file
# that sits next to it.
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("awq_to_ct", os.path.join(here, "awq-to-ct.py"))
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)


def awq_unpack_forward(packed: torch.Tensor) -> torch.Tensor:
    """Forward: slot s (bits 4s..4s+3) of column j goes to out-index j*8 + AWQ_ORDER[s]."""
    rows, cols = packed.shape
    out = torch.zeros(rows, cols * 8, dtype=torch.int32)
    for slot, pos in enumerate(AWQ_ORDER):
        out[:, pos::8] = (packed >> (4 * slot)) & 0xF
    return out


def awq_dequant(qweight, qzeros, scales, group_size) -> torch.Tensor:
    """w[o, i] = (q[i, o] - z[i // g, o]) * s[i // g, o]; the result is [out, in] fp16."""
    q = awq_unpack_forward(qweight)  # [in, out]
    z = awq_unpack_forward(qzeros).repeat_interleave(group_size, dim=0)
    s = scales.repeat_interleave(group_size, dim=0)
    return ((q - z).to(torch.float16) * s).t().contiguous()


def ct_dequant(fields: dict, group_size: int) -> torch.Tensor:
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4, type="int", symmetric=False, strategy="group", group_size=group_size
        ),
    )
    state = {k: fields[k] for k in FIELDS}
    return PackedQuantizationCompressor.decompress(state, scheme)["weight"]


class Reader:
    def __init__(self, root: str):
        self.root = root
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]
        self.handles = {}

    def has(self, name: str) -> bool:
        shard = self.weight_map.get(name)
        return shard is not None and os.path.exists(os.path.join(self.root, shard))

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(os.path.join(self.root, shard), framework="pt")
        return self.handles[shard].get_tensor(name)


def pick_modules(src: Reader, ignore: set[str]) -> list[str]:
    """Quantized modules of different kinds and layers; skips those whose shards are not there yet."""
    quantized = sorted(k[: -len(".qweight")] for k in src.weight_map if k.endswith(".qweight"))
    wanted = []
    for layer in (0, 1, 9, 20, 39):
        for expert in (0, 77, 255):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                wanted.append(f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}")
    for expert in (0, 5, 255):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            wanted.append(f"mtp.layers.0.mlp.experts.{expert}.{proj}")
    present = set(quantized)
    picked = [m for m in wanted if m in present and m not in ignore]
    return [m for m in picked if all(src.has(f"{m}.{s}") for s in ("qweight", "qzeros", "scales"))]


def check_numbers(src, dst, modules, group_size) -> int:
    failures = 0
    for module in modules:
        qw, qz, sc = (src.get(f"{module}.{s}") for s in ("qweight", "qzeros", "scales"))
        reference = awq_dequant(qw, qz, sc, group_size)
        sources = {"memory": conv.convert_module(qw, qz, sc, group_size)}
        if dst is not None and all(dst.has(f"{module}.{f}") for f in FIELDS):
            sources["output file"] = {f: dst.get(f"{module}.{f}") for f in FIELDS}
        for label, fields in sources.items():
            ok = fields["weight_packed"].dtype == torch.int32 and fields["weight_zero_point"].dtype == torch.int32
            got = ct_dequant(fields, group_size)
            ok = ok and got.dtype == torch.float16 and got.shape == reference.shape and torch.equal(got, reference)
            failures += not ok
            print(f"{'OK  ' if ok else 'FAIL'} {label:12s} {module} "
                  f"{tuple(reference.shape)} packed{tuple(fields['weight_packed'].shape)} "
                  f"zp{tuple(fields['weight_zero_point'].shape)}")
    return failures


def check_names(src, dst, ignore_set) -> int:
    expected = set()
    quantized = {k[: -len(".qweight")] for k in src.weight_map if k.endswith(".qweight")}
    for name in src.weight_map:
        if name.endswith((".qweight", ".qzeros", ".scales")):
            continue
        expected.add(name)
    for module in quantized:
        expected.update(f"{module}.{f}" for f in FIELDS)
    # Every quantized module must have all three input fields and must not be in ignore.
    bad = [m for m in quantized
           if m in ignore_set or f"{m}.qzeros" not in src.weight_map or f"{m}.scales" not in src.weight_map]
    got = set(dst.weight_map)
    missing, extra = sorted(expected - got), sorted(got - expected)
    print(f"names: expected {len(expected)}, in output {len(got)}, quantized modules {len(quantized)}, "
          f"missing {len(missing)}, extra {len(extra)}, bad input modules {len(bad)}")
    for n in (missing[:5] + extra[:5]):
        print("  ", n)
    # The index must match the real headers of the shards.
    on_disk = set()
    for shard in sorted(set(dst.weight_map.values())):
        with safe_open(os.path.join(dst.root, shard), framework="pt") as f:
            on_disk.update(f.keys())
    print(f"names in the shard headers: {len(on_disk)}, mismatch with the index: "
          f"{len(on_disk ^ got)}")
    return len(missing) + len(extra) + len(bad) + len(on_disk ^ got)


def check_copies(src, dst, ignore_set) -> int:
    """Non-quantized tensors of different kinds are bit-identical to the input."""
    patterns = [
        r"model\.language_model\.embed_tokens\.weight", r"lm_head\.weight",
        r"model\.language_model\.norm\.weight",
        r"model\.language_model\.layers\.0\.mlp\.gate\.weight",
        r"model\.language_model\.layers\.0\.mlp\.shared_expert\.(gate|up|down)_proj\.weight",
        r"model\.language_model\.layers\.0\.mlp\.shared_expert_gate\.weight",
        r"model\.language_model\.layers\.0\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj|conv1d|A_log|dt_bias|norm)\.weight|model\.language_model\.layers\.0\.linear_attn\.(A_log|dt_bias)",
        r"model\.language_model\.layers\.3\.self_attn\.(q|k|v|o)_proj\.weight",
        r"model\.language_model\.layers\.3\.self_attn\.(q|k)_norm\.weight",
        r"model\.language_model\.layers\.39\.(input|post_attention)_layernorm\.weight",
        r"mtp\.fc\.weight", r"mtp\.layers\.0\.self_attn\.q_proj\.weight",
        r"mtp\.layers\.0\.mlp\.gate\.weight", r"mtp\.layers\.0\.mlp\.shared_expert\.down_proj\.weight",
        r"mtp\.norm\.weight", r"model\.visual\.blocks\.0\.attn\.qkv\.weight",
        r"model\.visual\.merger\.linear_fc1\.weight",
    ]
    names = [n for n in src.weight_map if any(re.fullmatch(p, n) for p in patterns)]
    failures = 0
    for name in sorted(names):
        if not (src.has(name) and dst.has(name)):
            continue
        a, b = src.get(name), dst.get(name)
        ok = a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        failures += not ok
        print(f"{'OK  ' if ok else 'FAIL'} copy {name} {a.dtype} {tuple(a.shape)}")
    print(f"copies checked: {len(names)}")
    return failures


def check_config(src_dir, dst_dir, group_size) -> int:
    with open(os.path.join(src_dir, "config.json")) as f:
        src_cfg = json.load(f)
    with open(os.path.join(dst_dir, "config.json")) as f:
        dst_cfg = json.load(f)
    qc = dst_cfg["quantization_config"]
    QuantizationConfig.model_validate(qc)
    weights = qc["config_groups"]["group_0"]["weights"]
    ok = (qc["ignore"] == src_cfg["quantization_config"]["modules_to_not_convert"]
          and weights["symmetric"] is False and weights["group_size"] == group_size
          and weights["num_bits"] == 4 and qc["format"] == "pack-quantized")
    rest_ok = {k: v for k, v in src_cfg.items() if k != "quantization_config"} == \
              {k: v for k, v in dst_cfg.items() if k != "quantization_config"}
    print(f"config: QuantizationConfig valid, ignore/symmetric/group match: {ok}, "
          f"rest of config.json unchanged: {rest_ok}")
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(dst_dir)
        print("config: transformers.AutoConfig read it")
    except Exception as exc:  # report the cause; never skip silently
        print(f"FAIL AutoConfig: {exc!r}")
        return 1
    return 0 if ok and rest_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", required=True, help="AWQ checkpoint directory (input)")
    parser.add_argument("--dst", help="converted directory to check (optional)")
    parser.add_argument("--min-modules", type=int, default=20,
                        help="fewest modules the numeric check must cover (default 20)")
    args = parser.parse_args()

    with open(os.path.join(args.src, "config.json")) as f:
        awq = json.load(f)["quantization_config"]
    group_size = awq["group_size"]
    ignore_set = set(awq["modules_to_not_convert"])

    src = Reader(args.src)
    dst = Reader(args.dst) if args.dst and os.path.exists(
        os.path.join(args.dst, "model.safetensors.index.json")) else None

    modules = pick_modules(src, ignore_set)
    print(f"modules for the numeric check: {len(modules)}")
    failures = 0
    if len(modules) < args.min_modules:
        print(f"FAIL: at least {args.min_modules} modules are needed")
        failures += 1
    failures += check_numbers(src, dst, modules, group_size)
    if dst is not None:
        failures += check_names(src, dst, ignore_set)
        failures += check_copies(src, dst, ignore_set)
        failures += check_config(args.src, args.dst, group_size)
    else:
        print("output not created yet: names, copies and config checks skipped")
    print("SUMMARY:", "ALL OK" if failures == 0 else f"FAILURES: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
