#!/usr/bin/env python3
"""
Simple LoRA block weight merger for .safetensors files.

Example:
  python merge_lora.py \
    --a hina_Krea2Turbo_asianMix_v1_Lora.safetensors --wa 0.3 \
    --b hina_Krea2Raw_asianMix_v1_Lora.safetensors --wb 0.6 \
    --out hina_Krea2_merged_0p3_0p6.safetensors

By default, keys that exist in both LoRAs are merged as:
  out[key] = A[key] * wa + B[key] * wb

If one file has extra keys, the default is to error so you don't silently make a
broken LoRA. Use --missing copy/scale/skip if you intentionally want that.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

TOOL_VERSION = "krea2-keymap-raw-v3"

DTYPE_SIZES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "U64": 8,
    "I32": 4,
    "U32": 4,
    "I16": 2,
    "U16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


def die(msg: str, code: int = 2) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def fmt_keys(keys: Iterable[str], limit: int = 12) -> str:
    keys = sorted(keys)
    shown = "\n".join(f"  - {k}" for k in keys[:limit])
    if len(keys) > limit:
        shown += f"\n  ... and {len(keys) - limit} more"
    return shown or "  (none)"


KREA2_KEY_MAP_CHOICES = ("none", "krea2", "krea2-diffusion", "krea2-musubi")


def normalize_key_map_mode(mode: str) -> str:
    """Normalize key-map aliases to internal modes.

    ``krea2`` and ``krea2-diffusion`` convert musubi/kohya-style
    ``lora_unet_*`` keys into ``diffusion_model.*`` keys. ``krea2-musubi``
    does the inverse. ``none`` preserves input keys exactly.
    """
    if mode in {"none", ""}:
        return "none"
    if mode in {"krea2", "krea2-diffusion"}:
        return "krea2-diffusion"
    if mode == "krea2-musubi":
        return "krea2-musubi"
    die(f"unsupported --key-map value: {mode}")


def _split_lora_suffix(key: str) -> tuple[str, str] | None:
    if key.endswith(".lora_down.weight"):
        return key[: -len(".lora_down.weight")], ".lora_A.weight"
    if key.endswith(".lora_up.weight"):
        return key[: -len(".lora_up.weight")], ".lora_B.weight"
    if key.endswith(".lora_A.weight"):
        return key[: -len(".lora_A.weight")], ".lora_down.weight"
    if key.endswith(".lora_B.weight"):
        return key[: -len(".lora_B.weight")], ".lora_up.weight"
    if key.endswith(".alpha"):
        return key[: -len(".alpha")], ".alpha"
    return None


def krea2_musubi_to_diffusion_key(key: str) -> str:
    """Map musubi/kohya Krea2 LoRA keys to ``diffusion_model.*`` keys.

    Examples:
      lora_unet_blocks_9_mlp_gate.lora_down.weight
        -> diffusion_model.blocks.9.mlp.gate.lora_A.weight
      lora_unet_last_linear.alpha
        -> diffusion_model.last.linear.alpha
    """
    if not key.startswith("lora_unet_"):
        return key

    suffix_info = _split_lora_suffix(key)
    if suffix_info is None:
        return key
    body_with_prefix, suffix = suffix_info
    body = body_with_prefix[len("lora_unet_") :]

    patterns = [
        (
            r"blocks_(\d+)_(attn|mlp)_(wq|wk|wv|wo|gate|up|down)",
            "diffusion_model.blocks.{0}.{1}.{2}",
        ),
        (
            r"txtfusion_layerwise_blocks_(\d+)_(attn|mlp)_(wq|wk|wv|wo|gate|up|down)",
            "diffusion_model.txtfusion.layerwise_blocks.{0}.{1}.{2}",
        ),
        (
            r"txtfusion_refiner_blocks_(\d+)_(attn|mlp)_(wq|wk|wv|wo|gate|up|down)",
            "diffusion_model.txtfusion.refiner_blocks.{0}.{1}.{2}",
        ),
        (r"tmlp_(\d+)", "diffusion_model.tmlp.{0}"),
        (r"txtmlp_(\d+)", "diffusion_model.txtmlp.{0}"),
        (r"tproj_(\d+)", "diffusion_model.tproj.{0}"),
    ]

    for pattern, fmt in patterns:
        m = re.fullmatch(pattern, body)
        if m:
            return fmt.format(*m.groups()) + suffix

    special = {
        "first": "diffusion_model.first",
        "last_linear": "diffusion_model.last.linear",
        "txtfusion_projector": "diffusion_model.txtfusion.projector",
    }
    if body in special:
        return special[body] + suffix

    return key


def krea2_diffusion_to_musubi_key(key: str) -> str:
    """Map ``diffusion_model.*`` Krea2 LoRA keys to musubi/kohya-style keys."""
    if not key.startswith("diffusion_model."):
        return key

    suffix_info = _split_lora_suffix(key)
    if suffix_info is None:
        return key
    body_with_prefix, suffix = suffix_info
    body = body_with_prefix[len("diffusion_model.") :]

    patterns = [
        (
            r"blocks\.(\d+)\.(attn|mlp)\.(wq|wk|wv|wo|gate|up|down)",
            "lora_unet_blocks_{0}_{1}_{2}",
        ),
        (
            r"txtfusion\.layerwise_blocks\.(\d+)\.(attn|mlp)\.(wq|wk|wv|wo|gate|up|down)",
            "lora_unet_txtfusion_layerwise_blocks_{0}_{1}_{2}",
        ),
        (
            r"txtfusion\.refiner_blocks\.(\d+)\.(attn|mlp)\.(wq|wk|wv|wo|gate|up|down)",
            "lora_unet_txtfusion_refiner_blocks_{0}_{1}_{2}",
        ),
        (r"tmlp\.(\d+)", "lora_unet_tmlp_{0}"),
        (r"txtmlp\.(\d+)", "lora_unet_txtmlp_{0}"),
        (r"tproj\.(\d+)", "lora_unet_tproj_{0}"),
    ]

    for pattern, fmt in patterns:
        m = re.fullmatch(pattern, body)
        if m:
            return fmt.format(*m.groups()) + suffix

    special = {
        "first": "lora_unet_first",
        "last.linear": "lora_unet_last_linear",
        "txtfusion.projector": "lora_unet_txtfusion_projector",
    }
    if body in special:
        return special[body] + suffix

    return key


def map_krea2_lora_key(key: str, mode: str) -> str:
    mode = normalize_key_map_mode(mode)
    if mode == "none":
        return key
    if mode == "krea2-diffusion":
        return krea2_musubi_to_diffusion_key(key)
    if mode == "krea2-musubi":
        return krea2_diffusion_to_musubi_key(key)
    die(f"internal error: unknown key-map mode {mode!r}")


def remap_keyed_dict(items: Dict[str, Any], mode: str, label: str) -> Dict[str, Any]:
    mode = normalize_key_map_mode(mode)
    if mode == "none":
        return items

    out: Dict[str, Any] = {}
    changed = 0
    for key, value in items.items():
        new_key = map_krea2_lora_key(key, mode)
        changed += new_key != key
        if new_key in out:
            die(f"--key-map {mode} creates duplicate key for {label}: {new_key}")
        out[new_key] = value
    print(f"key-map {mode}: remapped {changed}/{len(items)} tensor keys for {label}")
    return out


def remap_header_keys(header: Dict[str, Any], mode: str, label: str) -> Dict[str, Any]:
    mode = normalize_key_map_mode(mode)
    if mode == "none":
        return header

    mapped: Dict[str, Any] = {}
    metadata = header.get("__metadata__", None)
    if metadata is not None:
        mapped["__metadata__"] = metadata
    mapped.update(remap_keyed_dict(tensor_items(header), mode, label))
    return mapped


def remap_header_keys_with_sources(header: Dict[str, Any], mode: str, label: str, *, verbose: bool = True) -> tuple[Dict[str, Any], Dict[str, str]]:
    """Return a remapped safetensors header and map output key -> original key.

    The raw merge path reads tensor bytes from the original safetensors file, but
    writes the mapped key names to the output. This reverse map lets us validate
    and write mapped keys without loading tensors through torch/numpy.
    """
    mode = normalize_key_map_mode(mode)
    if mode == "none":
        return header, {key: key for key in tensor_items(header)}

    mapped: Dict[str, Any] = {}
    source_by_mapped_key: Dict[str, str] = {}
    metadata = header.get("__metadata__", None)
    if metadata is not None:
        mapped["__metadata__"] = metadata

    changed = 0
    for old_key, meta in tensor_items(header).items():
        new_key = map_krea2_lora_key(old_key, mode)
        changed += new_key != old_key
        if new_key in mapped:
            die(f"--key-map {mode} creates duplicate key for {label}: {new_key}")
        mapped[new_key] = meta
        source_by_mapped_key[new_key] = old_key

    if verbose:
        print(f"key-map {mode}: remapped {changed}/{len(source_by_mapped_key)} tensor keys for {label}")
    return mapped, source_by_mapped_key


def normalize_block_key(key: str) -> str:
    """Normalize LoRA tensor keys for block filters.

    Filters may omit the leading ``diffusion_model.`` prefix. When the key is
    musubi/kohya-style, it is first converted to the equivalent diffusion-style
    key so filters such as ``blocks.*.attn.wk`` still work.
    """
    return krea2_musubi_to_diffusion_key(key).removeprefix("diffusion_model.")


def compile_block_patterns(blocks: str | None):
    """Compile comma-separated block patterns.

    Patterns are shell-like globs matched against both the full tensor key and a
    normalized key without the leading ``diffusion_model.`` prefix. ``*`` means
    any characters, while dots are literal. This makes examples like
    ``blocks.*.attn.wk,txtfusion.*.attn.gate.`` work naturally.
    """
    if not blocks:
        return []
    compiled = []
    for raw in blocks.split(","):
        pattern = raw.strip()
        if not pattern:
            continue
        regex = re.escape(pattern).replace(r"\*", ".*")
        try:
            compiled.append((pattern, re.compile(regex)))
        except re.error as exc:
            die(f"invalid --blocks pattern {pattern!r}: {exc}")
    return compiled


def key_matches_blocks(key: str, block_patterns) -> bool:
    if not block_patterns:
        return True
    normalized = normalize_block_key(key)
    return any(regex.match(key) or regex.match(normalized) for _, regex in block_patterns)


def selected_keys(keys: Iterable[str], block_patterns) -> set[str]:
    return {key for key in keys if key_matches_blocks(key, block_patterns)}


def read_safetensors_header(path: Path) -> Dict[str, Any]:
    """Read only the safetensors JSON header, no tensor data."""
    import struct

    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            die(f"{path} is too small to be a safetensors file")
        header_len = struct.unpack("<Q", raw_len)[0]
        header = f.read(header_len)
    try:
        return json.loads(header)
    except Exception as exc:
        die(f"failed to parse safetensors header for {path}: {exc}")


def tensor_items(header: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in header.items() if k != "__metadata__"}


def summarize(path: Path) -> Tuple[int, int, Dict[str, int]]:
    header = read_safetensors_header(path)
    tensors = tensor_items(header)
    dtypes: Dict[str, int] = {}
    for meta in tensors.values():
        dtype = meta.get("dtype", "UNKNOWN")
        dtypes[dtype] = dtypes.get(dtype, 0) + 1
    return len(tensors), len(header.get("__metadata__", {}) or {}), dtypes


def load_tensors(path: Path, backend: str, key_map: str = "none", label: str = "LoRA"):
    if backend == "torch":
        try:
            import torch  # noqa: F401
            from safetensors.torch import load_file
        except Exception as exc:
            die(f"torch backend requires torch + safetensors: {exc}")
        return remap_keyed_dict(load_file(str(path)), key_map, label), "torch"

    if backend == "numpy":
        try:
            from safetensors.numpy import load_file
        except Exception as exc:
            die(f"numpy backend requires numpy + safetensors: {exc}")
        return remap_keyed_dict(load_file(str(path)), key_map, label), "numpy"

    # auto: prefer torch because it handles BF16 and CUDA/CPU dtype casting well.
    try:
        import torch  # noqa: F401
        from safetensors.torch import load_file
        return remap_keyed_dict(load_file(str(path)), key_map, label), "torch"
    except Exception:
        pass

    try:
        from safetensors.numpy import load_file
        return remap_keyed_dict(load_file(str(path)), key_map, label), "numpy"
    except Exception as exc:
        die(
            "could not load safetensors. Install dependencies, e.g.\n"
            "  python -m pip install safetensors numpy\n"
            "For BF16 LoRAs, install torch too.\n"
            f"Original import error: {exc}"
        )


def save_tensors(tensors, metadata: Dict[str, str] | None, out: Path, backend: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if backend == "torch":
        from safetensors.torch import save_file
        save_file(tensors, str(out), metadata=metadata)
    elif backend == "numpy":
        from safetensors.numpy import save_file
        save_file(tensors, str(out), metadata=metadata)
    else:
        die(f"internal error: unknown backend {backend}")


def normalize_out_dtype(out_dtype: str) -> str:
    aliases = {
        "float32": "fp32",
        "f32": "fp32",
        "bfloat16": "bf16",
        "bf16": "bf16",
        "float16": "fp16",
        "f16": "fp16",
        "fp16": "fp16",
        "a": "a",
        "b": "b",
    }
    return aliases.get(out_dtype, out_dtype)


def cast_tensor(value, target: str, *, ref=None):
    """Cast torch Tensor or numpy ndarray to the requested output dtype."""
    target = normalize_out_dtype(target)
    if target in {"a", "b"}:
        if ref is None:
            die(f"internal error: output dtype {target!r} requires a reference tensor")
        if hasattr(value, "to") and hasattr(ref, "dtype"):
            return value.to(dtype=ref.dtype)
        return value.astype(ref.dtype, copy=False)

    if hasattr(value, "to"):
        import torch

        if target == "fp32":
            return value.to(dtype=torch.float32)
        if target == "bf16":
            return value.to(dtype=torch.bfloat16)
        if target == "fp16":
            return value.to(dtype=torch.float16)
        die(f"unsupported output dtype: {target}")

    if target == "fp32":
        return value.astype("float32")
    if target == "fp16":
        return value.astype("float16")
    if target == "bf16":
        die("--out-dtype bf16 requires the torch backend; numpy has no native bfloat16 dtype")
    die(f"unsupported output dtype: {target}")


def merge_tensors(a_tensors, b_tensors, wa: float, wb: float, missing: str, out_dtype: str, block_patterns=None, blocks_copy_from: str | None = None):
    a_keys = set(a_tensors.keys())
    b_keys = set(b_tensors.keys())
    common = sorted(a_keys & b_keys)
    only_a = sorted(a_keys - b_keys)
    only_b = sorted(b_keys - a_keys)

    if not common:
        die("A and B have no matching tensor keys; refusing to create an empty merge")

    if (only_a or only_b) and missing == "error":
        die(
            "LoRAs do not have identical tensor keys.\n"
            f"Only in A ({len(only_a)}):\n{fmt_keys(only_a)}\n"
            f"Only in B ({len(only_b)}):\n{fmt_keys(only_b)}\n"
            "Use --missing copy, --missing scale, or --missing skip if this is expected."
        )

    out = {}
    block_patterns = block_patterns or []
    active_keys = selected_keys([*common, *only_a, *only_b], block_patterns)

    def zero_like(value):
        if hasattr(value, "float"):
            return value.float() * 0.0
        return value.astype("float32") * 0.0

    def inactive_common_value(key: str):
        if blocks_copy_from == "a":
            return a_tensors[key]
        if blocks_copy_from == "b":
            return b_tensors[key]
        return zero_like(a_tensors[key])

    for key in common:
        av = a_tensors[key]
        bv = b_tensors[key]
        if tuple(av.shape) != tuple(bv.shape):
            die(f"shape mismatch for key {key}: A{tuple(av.shape)} vs B{tuple(bv.shape)}")
        if key in active_keys:
            merged = av.float() * wa + bv.float() * wb if hasattr(av, "float") else av.astype("float32") * wa + bv.astype("float32") * wb
        else:
            merged = inactive_common_value(key)
        ref = b_tensors[key] if normalize_out_dtype(out_dtype) == "b" else a_tensors[key]
        out[key] = cast_tensor(merged, out_dtype, ref=ref)

    if missing in {"copy", "scale"}:
        for key in only_a:
            v = a_tensors[key]
            if key in active_keys:
                merged = v.float() * wa if missing == "scale" and hasattr(v, "float") else v.astype("float32") * wa if missing == "scale" else v
            else:
                merged = zero_like(v)
            out[key] = cast_tensor(merged, out_dtype, ref=v)
        for key in only_b:
            v = b_tensors[key]
            if key in active_keys:
                merged = v.float() * wb if missing == "scale" and hasattr(v, "float") else v.astype("float32") * wb if missing == "scale" else v
            else:
                merged = zero_like(v)
            out[key] = cast_tensor(merged, out_dtype, ref=v)

    return out, common, only_a, only_b


def read_data_region(path: Path) -> tuple[Dict[str, Any], int]:
    """Return header and absolute byte offset where tensor data starts."""
    import struct

    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            die(f"{path} is too small to be a safetensors file")
        header_len = struct.unpack("<Q", raw_len)[0]
        header = json.loads(f.read(header_len))
        return header, 8 + header_len


def validate_raw_compatible(a_header: Dict[str, Any], b_header: Dict[str, Any], missing: str) -> tuple[list[str], list[str], list[str]]:
    a_items = tensor_items(a_header)
    b_items = tensor_items(b_header)
    a_keys = set(a_items)
    b_keys = set(b_items)
    common = sorted(a_keys & b_keys)
    only_a = sorted(a_keys - b_keys)
    only_b = sorted(b_keys - a_keys)

    if not common:
        die("A and B have no matching tensor keys; refusing to create an empty merge")
    if (only_a or only_b) and missing == "error":
        die(
            "LoRAs do not have identical tensor keys.\n"
            f"Only in A ({len(only_a)}):\n{fmt_keys(only_a)}\n"
            f"Only in B ({len(only_b)}):\n{fmt_keys(only_b)}\n"
            "Use --missing copy, --missing scale, or --missing skip if this is expected."
        )
    for key in common:
        am = a_items[key]
        bm = b_items[key]
        if am.get("shape") != bm.get("shape"):
            die(f"shape mismatch for key {key}: A{am.get('shape')} vs B{bm.get('shape')}")
    return common, only_a, only_b


def is_all_bf16(header: Dict[str, Any], keys: Iterable[str] | None = None) -> bool:
    items = tensor_items(header)
    selected = items if keys is None else {k: items[k] for k in keys}
    return bool(selected) and all(v.get("dtype") == "BF16" for v in selected.values())


def bf16_bytes_to_f32(raw: bytes):
    import numpy as np

    u16 = np.frombuffer(raw, dtype="<u2")
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def raw_float_bytes_to_f32(raw: bytes, dtype: str):
    import numpy as np

    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32)
    if dtype == "BF16":
        return bf16_bytes_to_f32(raw)
    die(f"raw float merge does not support dtype {dtype}")


def f32_to_bf16_bytes(values) -> bytes:
    import numpy as np

    f32 = np.asarray(values, dtype=np.float32)
    u32 = f32.view(np.uint32)
    # Round-to-nearest-even before truncating lower 16 bits.
    lsb = (u32 >> 16) & 1
    bias = np.uint32(0x7FFF) + lsb
    bf16 = ((u32 + bias) >> 16).astype("<u2", copy=False)
    return bf16.tobytes(order="C")


def f32_to_raw_float_bytes(values, dtype: str) -> bytes:
    import numpy as np

    if dtype == "F32":
        return np.asarray(values, dtype="<f4").tobytes(order="C")
    if dtype == "F16":
        return np.asarray(values, dtype="<f2").tobytes(order="C")
    if dtype == "BF16":
        return f32_to_bf16_bytes(values)
    die(f"raw float merge does not support output dtype {dtype}")


def safetensors_dtype_for_out(out_dtype: str, ref_dtype: str) -> str:
    target = normalize_out_dtype(out_dtype)
    if target in {"a", "b"}:
        return ref_dtype
    if target == "fp32":
        return "F32"
    if target == "bf16":
        return "BF16"
    if target == "fp16":
        return "F16"
    die(f"unsupported output dtype: {target}")


def read_tensor_raw(handle, data_start: int, meta: Dict[str, Any]) -> bytes:
    start, end = meta["data_offsets"]
    handle.seek(data_start + start)
    raw = handle.read(end - start)
    if len(raw) != end - start:
        die("unexpected EOF while reading tensor data")
    return raw


def tensor_nbytes(meta: Dict[str, Any]) -> int:
    dtype = meta.get("dtype")
    if dtype not in DTYPE_SIZES:
        die(f"unsupported dtype in raw writer: {dtype}")
    n = 1
    for dim in meta.get("shape", []):
        n *= int(dim)
    return n * DTYPE_SIZES[dtype]


def write_safetensors_raw(out: Path, header_items: Dict[str, Any], metadata: Dict[str, str], raw_chunks: Iterable[tuple[str, bytes]]) -> None:
    """Write a safetensors file from tensor metadata and already-serialized raw chunks."""
    import struct

    out.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    final_header: Dict[str, Any] = {"__metadata__": metadata}
    chunk_list = []
    for key, raw in raw_chunks:
        meta = dict(header_items[key])
        expected = tensor_nbytes(meta)
        if len(raw) != expected:
            die(f"internal error: tensor {key} raw size {len(raw)} != expected {expected}")
        meta["data_offsets"] = [offset, offset + len(raw)]
        final_header[key] = meta
        offset += len(raw)
        chunk_list.append(raw)

    # safetensors expects the JSON header to start with { and may be padded with spaces.
    header_bytes = json.dumps(final_header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * pad
    with out.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for raw in chunk_list:
            f.write(raw)


def raw_float_merge_supported(header: Dict[str, Any], keys: Iterable[str]) -> bool:
    items = tensor_items(header)
    return all(items[k].get("dtype") in {"F32", "F16", "BF16"} for k in keys)


def merge_float_raw(
    a_path: Path,
    b_path: Path,
    out: Path,
    wa: float,
    wb: float,
    missing: str,
    out_dtype: str,
    metadata: Dict[str, str],
    block_patterns=None,
    blocks_copy_from: str | None = None,
    key_map: str = "none",
) -> tuple[int, int, int, str]:
    """Merge F32/F16/BF16 safetensors through float32 without torch.

    Supports Krea2 key remapping by validating/writing mapped output keys while
    reading tensor payloads from their original source keys.
    """
    a_header_orig, a_data_start = read_data_region(a_path)
    b_header_orig, b_data_start = read_data_region(b_path)
    a_header, a_source_key = remap_header_keys_with_sources(a_header_orig, key_map, "A", verbose=False)
    b_header, b_source_key = remap_header_keys_with_sources(b_header_orig, key_map, "B", verbose=False)

    common, only_a, only_b = validate_raw_compatible(a_header, b_header, missing)
    if not raw_float_merge_supported(a_header, common) or not raw_float_merge_supported(b_header, common):
        die("raw float backend only supports F32, F16, and BF16 tensors")
    if missing in {"copy", "scale"}:
        if not raw_float_merge_supported(a_header, only_a) or not raw_float_merge_supported(b_header, only_b):
            die("raw float backend only supports F32, F16, and BF16 tensors")

    # Items keyed by output/mapped names for validation and output header.
    a_items = tensor_items(a_header)
    b_items = tensor_items(b_header)
    # Items keyed by original names for raw byte reads.
    a_orig_items = tensor_items(a_header_orig)
    b_orig_items = tensor_items(b_header_orig)

    normalized = normalize_out_dtype(out_dtype)
    block_patterns = block_patterns or []
    active_keys = selected_keys([*common, *only_a, *only_b], block_patterns)

    header_source: Dict[str, Any] = {}
    for key in common:
        ref_dtype = b_items[key]["dtype"] if normalized == "b" else a_items[key]["dtype"]
        meta = dict(a_items[key])
        meta["dtype"] = safetensors_dtype_for_out(out_dtype, ref_dtype)
        header_source[key] = meta
    if missing in {"copy", "scale"}:
        for key in only_a:
            meta = dict(a_items[key])
            meta["dtype"] = safetensors_dtype_for_out(out_dtype, a_items[key]["dtype"])
            header_source[key] = meta
        for key in only_b:
            meta = dict(b_items[key])
            meta["dtype"] = safetensors_dtype_for_out(out_dtype, b_items[key]["dtype"])
            header_source[key] = meta

    def read_a(fa, mapped_key: str) -> bytes:
        old_key = a_source_key[mapped_key]
        return read_tensor_raw(fa, a_data_start, a_orig_items[old_key])

    def read_b(fb, mapped_key: str) -> bytes:
        old_key = b_source_key[mapped_key]
        return read_tensor_raw(fb, b_data_start, b_orig_items[old_key])

    def dtype_a(mapped_key: str) -> str:
        return a_items[mapped_key]["dtype"]

    def dtype_b(mapped_key: str) -> str:
        return b_items[mapped_key]["dtype"]

    def convert_or_passthrough_from_a(fa, key: str) -> bytes:
        raw = read_a(fa, key)
        if header_source[key]["dtype"] == dtype_a(key):
            return raw
        av = raw_float_bytes_to_f32(raw, dtype_a(key))
        return f32_to_raw_float_bytes(av, header_source[key]["dtype"])

    def convert_or_passthrough_from_b(fb, key: str) -> bytes:
        raw = read_b(fb, key)
        if header_source[key]["dtype"] == dtype_b(key):
            return raw
        bv = raw_float_bytes_to_f32(raw, dtype_b(key))
        return f32_to_raw_float_bytes(bv, header_source[key]["dtype"])

    def chunks():
        with a_path.open("rb") as fa, b_path.open("rb") as fb:
            for key in common:
                if key not in active_keys:
                    if blocks_copy_from == "a":
                        yield key, convert_or_passthrough_from_a(fa, key)
                    elif blocks_copy_from == "b":
                        yield key, convert_or_passthrough_from_b(fb, key)
                    else:
                        yield key, b"\0" * tensor_nbytes(header_source[key])
                    continue

                av = raw_float_bytes_to_f32(read_a(fa, key), dtype_a(key))
                bv = raw_float_bytes_to_f32(read_b(fb, key), dtype_b(key))
                yield key, f32_to_raw_float_bytes(av * wa + bv * wb, header_source[key]["dtype"])

            if missing == "copy":
                for key in only_a:
                    if key not in active_keys:
                        yield key, b"\0" * tensor_nbytes(header_source[key])
                    else:
                        yield key, convert_or_passthrough_from_a(fa, key)
                for key in only_b:
                    if key not in active_keys:
                        yield key, b"\0" * tensor_nbytes(header_source[key])
                    else:
                        yield key, convert_or_passthrough_from_b(fb, key)
            elif missing == "scale":
                for key in only_a:
                    if key not in active_keys:
                        yield key, b"\0" * tensor_nbytes(header_source[key])
                        continue
                    av = raw_float_bytes_to_f32(read_a(fa, key), dtype_a(key))
                    yield key, f32_to_raw_float_bytes(av * wa, header_source[key]["dtype"])
                for key in only_b:
                    if key not in active_keys:
                        yield key, b"\0" * tensor_nbytes(header_source[key])
                        continue
                    bv = raw_float_bytes_to_f32(read_b(fb, key), dtype_b(key))
                    yield key, f32_to_raw_float_bytes(bv * wb, header_source[key]["dtype"])

    write_safetensors_raw(out, header_source, metadata, chunks())
    return len(common), len(only_a), len(only_b), "raw-float-numpy"

def merge_bf16_raw(a_path: Path, b_path: Path, out: Path, wa: float, wb: float, missing: str, metadata: Dict[str, str]) -> tuple[int, int, int, str]:
    """Backward-compatible wrapper: merge BF16 tensors through the generalized raw float path."""
    return merge_float_raw(a_path, b_path, out, wa, wb, missing, "a", metadata)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge two LoRA .safetensors files by weighted tensor-key addition.")
    p.add_argument("--version", action="version", version=TOOL_VERSION)
    p.add_argument("--a", required=True, type=Path, help="LoRA A .safetensors path")
    p.add_argument("--b", required=True, type=Path, help="LoRA B .safetensors path")
    p.add_argument("--out", required=True, type=Path, help="Output LoRA C .safetensors path")
    p.add_argument("--wa", type=float, required=True, help="Weight multiplier for LoRA A, e.g. 0.3")
    p.add_argument("--wb", type=float, required=True, help="Weight multiplier for LoRA B, e.g. 0.6")
    p.add_argument("--missing", choices=["error", "copy", "scale", "skip"], default="error", help="How to handle keys present in only one LoRA. Default: error")
    p.add_argument(
        "--out-dtype",
        choices=["a", "b", "fp32", "float32", "f32", "bf16", "bfloat16", "fp16", "float16", "f16"],
        default="a",
        help="Output dtype for all written tensors. Use a/b to follow input A/B, or fp32/bf16/fp16. Default: A's dtype",
    )
    p.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto", help="Tensor backend. auto prefers torch, falls back to numpy")
    p.add_argument(
        "--key-map",
        choices=KREA2_KEY_MAP_CHOICES,
        default="none",
        help=(
            "Map Krea2 LoRA key formats before comparing/merging. "
            "none preserves keys. krea2/krea2-diffusion converts musubi lora_unet_* keys "
            "to diffusion_model.* keys. krea2-musubi converts diffusion_model.* keys to lora_unet_* keys. "
            "F32/F16/BF16 tensors are merged through the raw-float path, so BF16 works without torch."
        ),
    )
    p.add_argument(
        "--blocks",
        default=None,
        help=(
            "Comma-separated block glob filters to merge. Dots are literal, * is wildcard, "
            "matched against full keys and keys without diffusion_model. prefix. "
            "By default, unmatched tensors are written as zeros; add --blocks-copy-from a/b to copy them. "
            "Example: blocks.*.attn.wk,txtfusion.*.attn.gate."
        ),
    )
    p.add_argument(
        "--blocks-copy-from",
        choices=["a", "b"],
        default=None,
        help=(
            "Use with --blocks: tensors that do not match --blocks are copied from LoRA A or LoRA B "
            "instead of being written as zeros. Selected tensors are still merged with wa/wb."
        ),
    )
    p.add_argument("--metadata", action="append", default=[], metavar="KEY=VALUE", help="Extra safetensors metadata entry. Can be repeated")
    p.add_argument("--dry-run", action="store_true", help="Inspect key compatibility without writing output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"tool version: {TOOL_VERSION}")
    if args.blocks_copy_from and not args.blocks:
        die("--blocks-copy-from must be used together with --blocks")
    for label, path in [("A", args.a), ("B", args.b)]:
        if not path.exists():
            die(f"LoRA {label} not found: {path}")
        if path.resolve() == args.out.resolve():
            die("output path must not overwrite an input file")

    for label, path in [("A", args.a), ("B", args.b)]:
        count, meta_count, dtypes = summarize(path)
        print(f"{label}: {path}")
        print(f"  tensors: {count}, metadata entries: {meta_count}, dtypes: {dtypes}")

    key_map = normalize_key_map_mode(args.key_map)
    a_header_original = read_safetensors_header(args.a)
    b_header_original = read_safetensors_header(args.b)
    a_header = remap_header_keys(a_header_original, key_map, "A")
    b_header = remap_header_keys(b_header_original, key_map, "B")
    common, only_a, only_b = validate_raw_compatible(a_header, b_header, args.missing)
    block_patterns = compile_block_patterns(args.blocks)
    output_key_candidates = [*common, *(only_a if args.missing in {"copy", "scale"} else []), *(only_b if args.missing in {"copy", "scale"} else [])]
    active_keys = selected_keys(output_key_candidates, block_patterns)
    print(f"common tensors: {len(common)}")
    print(f"only in A: {len(only_a)}")
    print(f"only in B: {len(only_b)}")
    if block_patterns:
        passthrough_count = len(output_key_candidates) - len(active_keys)
        print("block filters: " + ", ".join(pattern for pattern, _ in block_patterns))
        print(f"selected tensors: {len(active_keys)}")
        if args.blocks_copy_from:
            print(f"passthrough tensors copied from {args.blocks_copy_from.upper()}: {passthrough_count}")
        else:
            print(f"zeroed tensors: {passthrough_count}")

    if args.dry_run:
        if only_a:
            print("sample only-in-A keys:\n" + fmt_keys(only_a))
        if only_b:
            print("sample only-in-B keys:\n" + fmt_keys(only_b))
        if block_patterns:
            print("sample selected keys:\n" + fmt_keys(active_keys))
            inactive = sorted(set(output_key_candidates) - active_keys)
            if args.blocks_copy_from:
                print(f"sample passthrough keys copied from {args.blocks_copy_from.upper()}:\n" + fmt_keys(inactive))
            else:
                print("sample zeroed keys:\n" + fmt_keys(inactive))
        print("dry-run: no file written")
        return

    metadata = {
        "merge_lora_tool": "merge_lora.py",
        "tool_version": TOOL_VERSION,
        "merge_formula": f"A*{args.wa}+B*{args.wb}",
        "source_a": str(args.a),
        "source_b": str(args.b),
        "missing_policy": args.missing,
        "out_dtype": normalize_out_dtype(args.out_dtype),
        "blocks": args.blocks or "<all>",
        "blocks_copy_from": args.blocks_copy_from or "<zero>",
        "key_map": key_map,
    }
    for item in args.metadata:
        if "=" not in item:
            die(f"--metadata must be KEY=VALUE, got: {item}")
        k, v = item.split("=", 1)
        metadata[k] = v

    # Dependency-light path: merge F32/F16/BF16 tensors by converting each tensor
    # to float32 for math, then writing the requested output dtype. This handles
    # mixed F32/BF16 LoRAs without requiring torch.
    normalized_out_dtype = normalize_out_dtype(args.out_dtype)
    raw_supported = raw_float_merge_supported(a_header, common) and raw_float_merge_supported(b_header, common)
    if args.backend in {"auto", "numpy"} and raw_supported:
        common_n, only_a_n, only_b_n, backend = merge_float_raw(args.a, args.b, args.out, args.wa, args.wb, args.missing, normalized_out_dtype, metadata, block_patterns, args.blocks_copy_from, key_map)
    else:
        a, backend = load_tensors(args.a, args.backend, key_map, "A")
        b, backend2 = load_tensors(args.b, backend, key_map, "B")
        if backend2 != backend:
            die("internal error: mixed backends")
        out_tensors, common, only_a, only_b = merge_tensors(a, b, args.wa, args.wb, args.missing, args.out_dtype, block_patterns, args.blocks_copy_from)
        save_tensors(out_tensors, metadata, args.out, backend)
        common_n, only_a_n, only_b_n = len(common), len(only_a), len(only_b)

    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"backend: {backend}")
    print(f"wrote: {args.out} ({size_mb:.2f} MiB)")
    print(f"merged common tensors: {common_n}")
    if args.missing != "error":
        print(f"handled missing tensors: A-only={only_a_n}, B-only={only_b_n}, policy={args.missing}")


if __name__ == "__main__":
    main()
