#!/usr/bin/env python3
"""Converts an AutoAWQ checkpoint (GEMM layout, with zero points) into the
compressed-tensors ``pack-quantized`` format (asymmetric, 4 bits).

Why: the RDNA2 MoE kernel of the vLLM fork exists only on the compressed-tensors
WNA16 path (compressed_tensors_moe_wna16_rdna2.py), where asymmetric zero points
are taken as they are (zero_offset=0). The conversion is lossless: it only repacks
the same 4-bit values, and the scales stay fp16 unchanged.

Correspondence of the formats (the source of truth is awq_triton.py in the fork and
compressed_tensors/compressors/pack_quantized/):

  AWQ                                  compressed-tensors
  qweight  int32 [in, out/8]           weight_packed     int32 [out, in/8]
           (packed along out, AWQ slot          (packed along in, slots 0..7
           order [0,2,4,6,1,3,5,7])              in natural order, value q)
  qzeros   int32 [in/g, out/8]         weight_zero_point int32 [out/8, in/g]
  scales   fp16  [in/g, out]           weight_scale      fp16  [out, in/g]
                                       weight_shape      int64 [out, in]

Why the values carry over without a shift: the library writes weight_packed and
weight_zero_point as int8 + 8 (offset = 1 << (bits - 1)), so the signed q - 8
becomes the unsigned q. Hence the unsigned nibbles of the output equal the AWQ
nibbles (q and z), and dequantization (q - z) * s matches bit for bit.

Run with the venv interpreter, $VLLM_RDNA2_HOME/venv/bin/python; no GPU is needed:
  awq-to-ct.py --src AWQ_DIR --dst OUTPUT_DIR
"""

import argparse
import json
import os
import shutil
import sys
import time

import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from safetensors import safe_open
from safetensors.torch import save_file

BITS = 4
PACK = 32 // BITS
# AWQ packs the out-index j*8+m into slot REVERSE_AWQ_ORDER[m] of each int32
# (the same order as in vllm/.../auto_awq.py and awq_triton.py).
REVERSE_AWQ_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

# Files that are not copied: the index and the shards are rebuilt, config.json is
# rewritten, and the rest describes the AWQ quantization, which would be stale
# next to the new checkpoint.
SKIP_FILES = {
    "model.safetensors.index.json",
    "config.json",
    "quant_config.json",
    "recipe.yaml",
    "README.md",
    ".gitattributes",
}


def unpack_awq(packed: torch.Tensor) -> torch.Tensor:
    """int32 [R, C/8] in AWQ order -> int32 [R, C] with values 0..15."""
    shifts = torch.arange(0, 32, BITS, dtype=torch.int32)
    nibbles = (packed.unsqueeze(-1) >> shifts) & ((1 << BITS) - 1)
    nibbles = nibbles[:, :, REVERSE_AWQ_ORDER]
    return nibbles.reshape(packed.shape[0], packed.shape[1] * PACK)


def convert_module(
    qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, group_size: int
) -> dict[str, torch.Tensor]:
    """One quantized AWQ module -> pack-quantized fields (without the module prefix)."""
    in_features, out_packed = qweight.shape
    out_features = out_packed * PACK
    num_groups = scales.shape[0]
    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise ValueError(f"qweight and qzeros must be int32: {qweight.dtype}, {qzeros.dtype}")
    # bf16 would change the values when converted to fp16 and break bit-exact equality.
    if scales.dtype != torch.float16:
        raise ValueError(f"scales must be fp16, not {scales.dtype}")
    if in_features % group_size or num_groups != in_features // group_size:
        raise ValueError(f"group {group_size} does not match: in={in_features}, groups={num_groups}")
    if scales.shape != (num_groups, out_features) or qzeros.shape != (num_groups, out_packed):
        raise ValueError(f"shapes: qzeros {tuple(qzeros.shape)}, scales {tuple(scales.shape)}")

    offset = 1 << (BITS - 1)
    q = unpack_awq(qweight)  # [in, out]
    z = unpack_awq(qzeros)  # [groups, out]
    weight_packed = pack_to_int32((q.t() - offset).to(torch.int8).contiguous(), BITS)
    # packed_dim=0: packing along out, result [out/8, groups] (as the compressor does).
    zero_point = pack_to_int32((z.t() - offset).to(torch.int8).contiguous(), BITS, packed_dim=0)
    return {
        "weight_packed": weight_packed.contiguous(),
        "weight_scale": scales.t().contiguous(),
        "weight_shape": torch.tensor([out_features, in_features], dtype=torch.int64),
        "weight_zero_point": zero_point.contiguous(),
    }


def build_quantization_config(ignore: list[str], group_size: int) -> dict:
    """Structure taken from the sample Qwen3.5-35B-A3B-AWQ-4bit; only symmetric is false."""
    return {
        "config_groups": {
            "group_0": {
                "format": "pack-quantized",
                "input_activations": None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": group_size,
                    "num_bits": BITS,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "scale_dtype": None,
                    "strategy": "group",
                    "symmetric": False,
                    "type": "int",
                    "zp_dtype": None,
                },
            }
        },
        "format": "pack-quantized",
        "global_compression_ratio": None,
        "ignore": ignore,
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "sparsity_config": {},
        "transform_config": {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", required=True, help="AWQ checkpoint directory (input)")
    parser.add_argument("--dst", required=True, help="output directory (created if missing)")
    args = parser.parse_args()

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

    with open(os.path.join(args.src, "config.json")) as f:
        config = json.load(f)
    awq = config["quantization_config"]
    if not (awq.get("quant_method") == "awq" and awq.get("version") == "gemm"
            and awq.get("zero_point") is True and awq.get("bits") == BITS):
        raise SystemExit(f"input is not an AWQ GEMM int4 checkpoint with zero points: {awq}")
    group_size = awq["group_size"]
    ignore = list(awq["modules_to_not_convert"])
    ignore_set = set(ignore)

    with open(os.path.join(args.src, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    src_shards = sorted(set(weight_map.values()))
    missing = [s for s in src_shards if not os.path.exists(os.path.join(args.src, s))]
    if missing:
        raise SystemExit(f"input shards are missing: {missing}")

    os.makedirs(args.dst, exist_ok=True)
    handles = {s: safe_open(os.path.join(args.src, s), framework="pt") for s in src_shards}

    def get(name: str) -> torch.Tensor:
        return handles[weight_map[name]].get_tensor(name)

    out_map: dict[str, str] = {}
    total_size = 0
    n_shards = len(src_shards)
    for idx, shard in enumerate(src_shards, 1):
        out_name = f"model-{idx:05d}-of-{n_shards:05d}.safetensors"
        out_path = os.path.join(args.dst, out_name)
        names = list(handles[shard].keys())
        # The output names are known in advance, so the index is built even for
        # shards that already exist.
        planned: dict[str, int] = {}
        t0 = time.time()
        tensors: dict[str, torch.Tensor] = {}
        for name in names:
            if name.endswith((".qzeros", ".scales")):
                continue  # the module that owns qweight consumes them
            if name.endswith(".qweight"):
                module = name[: -len(".qweight")]
                if module in ignore_set:
                    raise SystemExit(f"{module} is in ignore but has qweight")
                for suffix in ("qzeros", "scales"):
                    if f"{module}.{suffix}" not in weight_map:
                        raise SystemExit(f"{module}: no {suffix}")
                if os.path.exists(out_path):
                    # The shard is already written (recovery after a failure): names only.
                    for field in ("weight_packed", "weight_scale", "weight_shape", "weight_zero_point"):
                        planned[f"{module}.{field}"] = 0
                    continue
                fields = convert_module(
                    get(name), get(f"{module}.qzeros"), get(f"{module}.scales"), group_size
                )
                for field, tensor in fields.items():
                    tensors[f"{module}.{field}"] = tensor
            else:
                if os.path.exists(out_path):
                    planned[name] = 0
                else:
                    tensors[name] = get(name).contiguous()
        if os.path.exists(out_path):
            print(f"[{idx}/{n_shards}] {out_name} already exists, skipped", flush=True)
            names_out = list(planned)
            total_size += os.path.getsize(out_path)  # approximate: includes the header
        else:
            names_out = list(tensors)
            total_size += sum(t.numel() * t.element_size() for t in tensors.values())
            tmp_path = out_path + ".tmp"
            save_file(tensors, tmp_path, metadata={"format": "pt"})
            os.replace(tmp_path, out_path)
            print(f"[{idx}/{n_shards}] {out_name}: {len(tensors)} tensors, "
                  f"{time.time() - t0:.0f} s", flush=True)
            del tensors
        for n in names_out:
            out_map[n] = out_name

    with open(os.path.join(args.dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": out_map}, f, indent=2)

    config["quantization_config"] = build_quantization_config(ignore, group_size)
    with open(os.path.join(args.dst, "config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    for fname in sorted(os.listdir(args.src)):
        path = os.path.join(args.src, fname)
        if fname in SKIP_FILES or fname.endswith(".safetensors") or not os.path.isfile(path):
            continue
        shutil.copyfile(path, os.path.join(args.dst, fname))
    print(f"DONE: {len(out_map)} tensors, {total_size / 1e9:.2f} GB -> {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
