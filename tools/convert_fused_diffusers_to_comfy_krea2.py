#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

BLOCK_LEAVES = {
    "attn.to_q": "attn.wq",
    "attn.to_k": "attn.wk",
    "attn.to_v": "attn.wv",
    "attn.to_gate": "attn.gate",
    "attn.to_out.0": "attn.wo",
    "ff.gate": "mlp.gate",
    "ff.up": "mlp.up",
    "ff.down": "mlp.down",
}
TOP_LINEARS = {
    "img_in": "first",
    "time_embed.linear_1": "tmlp.0",
    "time_embed.linear_2": "tmlp.2",
    "time_mod_proj": "tproj.1",
    "txt_in.linear_1": "txtmlp.1",
    "txt_in.linear_2": "txtmlp.3",
    "text_fusion.projector": "txtfusion.projector",
    "final_layer.linear": "last.linear",
}
EXACT = {
    "txt_in.norm.weight": "txtmlp.0.scale",
    "final_layer.norm.weight": "last.norm.scale",
    "final_layer.scale_shift_table": "last.modulation.lin",
}
BLOCK_EXACT = {
    "attn.norm_q.weight": "attn.qknorm.qnorm.scale",
    "attn.norm_k.weight": "attn.qknorm.knorm.scale",
    "norm1.weight": "prenorm.scale",
    "norm2.weight": "postnorm.scale",
    "scale_shift_table": "mod.lin",
}


def map_key(key: str):
    if key in EXACT:
        return EXACT[key], False
    for src, dst in TOP_LINEARS.items():
        for suffix in (".weight", ".bias"):
            if key == src + suffix:
                return dst + suffix, False

    m = re.match(r"^transformer_blocks\.(\d+)\.(.+)$", key)
    if m:
        block, leaf = m.groups()
        if leaf in BLOCK_EXACT:
            return f"blocks.{block}.{BLOCK_EXACT[leaf]}", leaf == "scale_shift_table"
        if leaf.endswith(".weight"):
            stem = leaf[:-7]
            if stem in BLOCK_LEAVES:
                return f"blocks.{block}.{BLOCK_LEAVES[stem]}.weight", False

    m = re.match(r"^text_fusion\.(layerwise_blocks|refiner_blocks)\.(\d+)\.(.+)$", key)
    if m:
        group, block, leaf = m.groups()
        if leaf in BLOCK_EXACT and leaf != "scale_shift_table":
            return f"txtfusion.{group}.{block}.{BLOCK_EXACT[leaf]}", False
        if leaf.endswith(".weight"):
            stem = leaf[:-7]
            if stem in BLOCK_LEAVES:
                return f"txtfusion.{group}.{block}.{BLOCK_LEAVES[stem]}.weight", False

    raise KeyError(f"unmapped Krea2 key: {key}")


def index_sources(src_dir: Path):
    index_path = src_dir / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    by_file = {}
    for key, filename in index["weight_map"].items():
        by_file.setdefault(filename, []).append(key)
    return by_file


def header_keys(path: Path, logical: bool = False):
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
    if not logical:
        return keys
    normalized = set()
    for key in keys:
        for suffix in (".weight_scale_2", ".weight_scale", ".comfy_quant"):
            if key.endswith(suffix):
                key = key[:-len(suffix)] + ".weight"
                break
        normalized.add(key)
    return normalized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--oracle", type=Path)
    ap.add_argument("--resume-tmp", action="store_true")
    args = ap.parse_args()

    tmp = Path(str(args.dst) + ".tmp")
    if not args.resume_tmp:
        tmp.unlink(missing_ok=True)
        out = {}
        source_count = 0
        for filename, keys in index_sources(args.src_dir).items():
            with safe_open(args.src_dir / filename, framework="pt", device="cpu") as f:
                for key in keys:
                    new_key, flatten = map_key(key)
                    if new_key in out:
                        raise RuntimeError(f"duplicate mapped key: {new_key}")
                    tensor = f.get_tensor(key)
                    if flatten:
                        tensor = tensor.reshape(-1)
                    out[new_key] = tensor
                    source_count += 1

        if source_count != 430 or len(out) != 430:
            raise RuntimeError(f"expected 430 tensors, source={source_count}, mapped={len(out)}")

        save_file(out, tmp, metadata={
            "key_layout": "comfyui_krea2",
            "converted_from": "diffusers_krea2",
            "source": str(args.src_dir),
        })
        del out
    elif not tmp.is_file():
        raise FileNotFoundError(f"resume requested but temp file is missing: {tmp}")

    actual = header_keys(tmp)
    required = {
        "txtfusion.projector.weight", "first.weight",
        "blocks.0.attn.wq.weight", "blocks.27.mlp.down.weight",
        "last.linear.weight",
    }
    if not required.issubset(actual):
        raise RuntimeError(f"missing signatures: {sorted(required - actual)}")
    if args.oracle:
        actual_logical = header_keys(tmp, logical=True)
        expected = header_keys(args.oracle, logical=True)
        if actual_logical != expected:
            raise RuntimeError(
                f"oracle mismatch: missing={sorted(expected-actual_logical)[:20]} "
                f"extra={sorted(actual_logical-expected)[:20]}"
            )

    os.replace(tmp, args.dst)
    print(json.dumps({
        "output": str(args.dst),
        "tensor_count": len(actual),
        "oracle_match": bool(args.oracle),
        "detection_signature": "txtfusion.projector.weight",
    }, indent=2))


if __name__ == "__main__":
    main()
