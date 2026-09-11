#!/usr/bin/env python3
"""
Krea2 checkpoint-difference -> LoRA converter.

Given two compatible Krea2 diffusion-model checkpoints A and B, computes
    delta = A - B
per matrix parameter and approximates delta with a low-rank factorization.
Applying the produced LoRA to checkpoint B at strength 1.0 approximates A.

Designed for single-file .safetensors checkpoints, including FP8 / scaled-FP8
files commonly used by ComfyUI.

Examples
--------
# Fixed rank 64, CUDA randomized SVD, BF16 LoRA output
python krea2_delta_to_lora.py \
  --a model_A_fp8.safetensors \
  --b model_B_fp8.safetensors \
  --output A_minus_B_rank64.safetensors \
  --rank 64 --device cuda

# Adaptive rank up to 256, target 98% Frobenius-energy capture
python krea2_delta_to_lora.py \
  --a A.safetensors --b B.safetensors \
  --output A_minus_B_adaptive.safetensors \
  --rank 256 --energy 0.98 --device cuda

Notes
-----
* The subtraction is always A - B.
* FP8 tensors are converted/dequantized to FP32 before subtraction/SVD.
* For scaled FP8, companion weight_scale tensors are applied when present.
* By default, only ComfyUI's canonical Krea2 LoRA-mapped Linear modules are targeted.
* Standard output is ComfyUI-friendly .lora_down/.lora_up keys with alpha=rank.
* Non-matrix parameters (bias, RMSNorm scales, 1D modulation vectors) are skipped
  unless --include-nonmatrix-diff is specified; that option creates a hybrid
  LoRA+exact-diff patch rather than a pure LoRA.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open
from safetensors.torch import save_file


QUANT_AUX_SUFFIXES = (
    ".weight_scale",
    ".weight_scale_2",
    ".pre_quant_scale",
    ".input_scale",
    ".scale_weight",
    ".comfy_quant",
)

SPECIAL_KEYS = {
    "scaled_fp8",
    "_quantization_metadata",
    "__metadata__",
}


@dataclass
class LayerReport:
    key: str
    shape: list[int]
    kind: str
    rank: int | None
    captured_energy: float | None
    delta_fro_norm: float
    relative_delta_norm: float
    scale_a: str | None = None
    scale_b: str | None = None
    note: str | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert A-B checkpoint difference into a Krea2/ComfyUI LoRA.",
    )
    p.add_argument("--a", required=True, help="Target/modified checkpoint A (.safetensors)")
    p.add_argument("--b", required=True, help="Base checkpoint B (.safetensors). Apply LoRA to B.")
    p.add_argument("--output", "-o", required=True, help="Output LoRA .safetensors")
    p.add_argument("--rank", type=int, default=64, help="Maximum/fixed rank")
    p.add_argument(
        "--energy",
        type=float,
        default=None,
        help="Adaptive rank target in (0,1], e.g. 0.98. Rank is capped by --rank.",
    )
    p.add_argument("--min-rank", type=int, default=1, help="Minimum adaptive rank")
    p.add_argument(
        "--method",
        choices=("auto", "lowrank", "exact"),
        default="auto",
        help="SVD method. auto uses exact SVD only for small matrices.",
    )
    p.add_argument("--oversample", type=int, default=16, help="Randomized SVD oversampling")
    p.add_argument("--niter", type=int, default=2, help="Randomized SVD power iterations")
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="SVD device, e.g. cuda, cuda:0, cpu",
    )
    p.add_argument(
        "--output-dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="LoRA tensor dtype",
    )
    p.add_argument(
        "--format",
        choices=("comfy-ab", "comfy-updown"),
        default="comfy-updown",
        help="LoRA tensor key suffix convention",
    )
    p.add_argument(
        "--prefix",
        choices=("auto", "diffusion_model", "none"),
        default="auto",
        help="Output target-key prefix policy",
    )
    p.add_argument(
        "--target-set",
        choices=("krea2", "all-matrix"),
        default="krea2",
        help=(
            "Which matrices may become LoRA targets. krea2 restricts output to the "
            "official Krea2 LoRA-mapped Linear modules; all-matrix is legacy behavior."
        ),
    )
    p.add_argument(
        "--warn-energy",
        type=float,
        default=0.90,
        help="Warn when a layer's rank-capped SVD captures less than this fraction of delta energy.",
    )
    p.add_argument(
        "--warn-relative-delta",
        type=float,
        default=0.25,
        help="Warn when ||A-B||/||B|| for a layer exceeds this value.",
    )
    p.add_argument(
        "--factor-balance",
        choices=("sqrt", "up"),
        default="sqrt",
        help="Split singular values between factors (sqrt is better conditioned)",
    )
    p.add_argument(
        "--include",
        action="append",
        default=[],
        help="Regex; only matching parameter keys are processed. Repeatable.",
    )
    p.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Regex; matching parameter keys are skipped. Repeatable.",
    )
    p.add_argument(
        "--include-nonmatrix-diff",
        action="store_true",
        help="Store exact .diff patches for changed non-2D parameters (hybrid patch).",
    )
    p.add_argument(
        "--nonmatrix-threshold",
        type=float,
        default=0.0,
        help="Minimum Frobenius norm for exact non-matrix diff patches.",
    )
    p.add_argument(
        "--min-delta-norm",
        type=float,
        default=0.0,
        help="Skip matrix deltas with Frobenius norm <= this value.",
    )
    p.add_argument("--seed", type=int, default=12345, help="Randomized SVD seed")
    p.add_argument(
        "--report",
        default=None,
        help="Optional JSON report path. Default: <output>.report.json",
    )
    p.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write a JSON report",
    )
    p.add_argument(
        "--strict-keys",
        action="store_true",
        help="Fail if A/B have different non-quantization parameter key sets.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect compatible keys/scales without performing SVD or writing LoRA.",
    )
    return p.parse_args()


def output_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def is_aux_key(key: str) -> bool:
    if key in SPECIAL_KEYS or key.startswith("__"):
        return True
    return any(key.endswith(s) for s in QUANT_AUX_SUFFIXES)


def passes_filters(key: str, include: list[re.Pattern], exclude: list[re.Pattern]) -> bool:
    if include and not any(r.search(key) for r in include):
        return False
    if exclude and any(r.search(key) for r in exclude):
        return False
    return True


def shape_of(f, key: str) -> tuple[int, ...]:
    return tuple(f.get_slice(key).get_shape())


def _strip_model_prefix(key: str) -> str:
    for p in ("model.diffusion_model.", "diffusion_model.", "transformer."):
        if key.startswith(p):
            return key[len(p):]
    return key


def krea2_canonical_lora_base(key: str) -> str | None:
    """Return canonical ComfyUI/Diffusers Krea2 LoRA base key, or None.

    This intentionally mirrors ComfyUI's krea2_to_diffusers mapping and does NOT
    treat arbitrary 2D parameters (e.g. modulation tables) as LoRA layers.
    """
    k = _strip_model_prefix(key)
    if not k.endswith(".weight"):
        return None
    k = k[:-len(".weight")]

    # Already in Krea2 Diffusers namespace.
    diffusers_roots = (
        "transformer_blocks.",
        "text_fusion.layerwise_blocks.",
        "text_fusion.refiner_blocks.",
    )
    if k.startswith(diffusers_roots):
        # Only allow the module families ComfyUI maps for Krea2.
        if re.search(r"\.attn\.(to_q|to_k|to_v|to_gate|to_out(?:\.0)?)$", k):
            if k.endswith(".attn.to_out"):
                k += ".0"
            return f"diffusion_model.{k}"
        if re.search(r"\.ff\.(gate|up|down)$", k):
            return f"diffusion_model.{k}"
        return None

    basic_diffusers = {
        "img_in",
        "time_embed.linear_1",
        "time_embed.linear_2",
        "time_mod_proj",
        "txt_in.linear_1",
        "txt_in.linear_2",
        "text_fusion.projector",
        "final_layer.linear",
    }
    if k in basic_diffusers:
        return f"diffusion_model.{k}"

    # Native Krea2 -> Diffusers names, matching comfy.utils.krea2_to_diffusers().
    module_map = {
        "attn.wq": "attn.to_q",
        "attn.wk": "attn.to_k",
        "attn.wv": "attn.to_v",
        "attn.gate": "attn.to_gate",
        "attn.wo": "attn.to_out.0",
        "mlp.gate": "ff.gate",
        "mlp.up": "ff.up",
        "mlp.down": "ff.down",
    }

    m = re.fullmatch(r"blocks\.(\d+)\.(attn\.(?:wq|wk|wv|gate|wo)|mlp\.(?:gate|up|down))", k)
    if m:
        i, mod = m.groups()
        return f"diffusion_model.transformer_blocks.{i}.{module_map[mod]}"

    m = re.fullmatch(
        r"txtfusion\.(layerwise_blocks|refiner_blocks)\.(\d+)\."
        r"(attn\.(?:wq|wk|wv|gate|wo)|mlp\.(?:gate|up|down))",
        k,
    )
    if m:
        group, i, mod = m.groups()
        return f"diffusion_model.text_fusion.{group}.{i}.{module_map[mod]}"

    basic_native = {
        "first": "img_in",
        "tmlp.0": "time_embed.linear_1",
        "tmlp.2": "time_embed.linear_2",
        "tproj.1": "time_mod_proj",
        "txtmlp.1": "txt_in.linear_1",
        "txtmlp.3": "txt_in.linear_2",
        "txtfusion.projector": "text_fusion.projector",
        "last.linear": "final_layer.linear",
    }
    mapped = basic_native.get(k)
    if mapped is not None:
        return f"diffusion_model.{mapped}"
    return None


def normalize_target_base(key: str, prefix_policy: str) -> str:
    """Legacy generic target mapping, used only with --target-set all-matrix."""
    if key.endswith(".weight"):
        base = key[:-len(".weight")]
    else:
        base = key
    if prefix_policy == "none":
        return base
    if prefix_policy == "diffusion_model":
        return base if base.startswith("diffusion_model.") else f"diffusion_model.{base}"
    if base.startswith("diffusion_model."):
        return base
    if base.startswith("model.diffusion_model."):
        return base[len("model."):]
    return f"diffusion_model.{base}"

def parse_quant_metadata(f) -> dict[str, Any]:
    md = f.metadata() or {}
    candidates = []
    for k in ("_quantization_metadata", "quantization_metadata", "metadata.json"):
        if k in md:
            candidates.append(md[k])
    # Some tools put one JSON object in an arbitrary metadata value.
    candidates.extend(v for v in md.values() if isinstance(v, str) and "quantization" in v.lower())

    for raw in candidates:
        try:
            obj = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            continue
        if isinstance(obj, dict) and "_quantization_metadata" in obj:
            obj = obj["_quantization_metadata"]
        if isinstance(obj, dict) and "layers" in obj:
            return obj
    return {}


def quant_format_for_key(qmeta: dict[str, Any], key: str) -> str | None:
    layers = qmeta.get("layers", {}) if isinstance(qmeta, dict) else {}
    layer_key = key[:-len(".weight")] if key.endswith(".weight") else key
    info = layers.get(layer_key)
    if isinstance(info, str):
        return info
    if isinstance(info, dict):
        fmt = info.get("format")
        return str(fmt) if fmt is not None else None
    return None


def reinterpret_quant_storage(t: torch.Tensor, fmt: str | None, key: str) -> torch.Tensor:
    """Handle checkpoints that store FP8 payload bytes in uint8 containers."""
    if t.dtype != torch.uint8:
        return t
    if fmt is None:
        raise TypeError(
            f"{key}: uint8 tensor has no recognized quantization metadata; "
            "cannot safely reinterpret/dequantize it."
        )
    fmt_l = fmt.lower()
    if "float8_e4m3" in fmt_l or "fp8_e4m3" in fmt_l or fmt_l in {"e4m3", "fp8"}:
        return t.contiguous().view(torch.float8_e4m3fn)
    if "float8_e5m2" in fmt_l or "fp8_e5m2" in fmt_l or fmt_l == "e5m2":
        return t.contiguous().view(torch.float8_e5m2)
    raise TypeError(f"{key}: unsupported uint8-backed quantization format {fmt!r}")


def find_scale_key(f, key: str) -> str | None:
    keys = f.keys()
    candidates: list[str] = []
    if key.endswith(".weight"):
        base = key[:-len(".weight")]
        candidates.extend(
            [
                f"{base}.weight_scale",
                f"{base}.scale_weight",
                f"{key}_scale",
            ]
        )
    else:
        candidates.extend([f"{key}_scale", f"{key}.weight_scale"])
    for c in candidates:
        if c in keys:
            return c
    return None


def broadcast_scale(scale: torch.Tensor, weight: torch.Tensor, key: str) -> torch.Tensor:
    scale = scale.to(device=weight.device, dtype=torch.float32)
    if scale.numel() == 1:
        return scale.reshape(())
    if tuple(scale.shape) == tuple(weight.shape):
        return scale
    # Common per-output-channel scale.
    if scale.ndim == 1 and weight.ndim >= 2 and scale.shape[0] == weight.shape[0]:
        return scale.reshape(scale.shape[0], *([1] * (weight.ndim - 1)))
    # Less common per-input-channel scale.
    if scale.ndim == 1 and weight.ndim >= 2 and scale.shape[0] == weight.shape[-1]:
        return scale.reshape(*([1] * (weight.ndim - 1)), scale.shape[0])
    try:
        torch.broadcast_shapes(scale.shape, weight.shape)
        return scale
    except RuntimeError as e:
        raise ValueError(
            f"{key}: scale shape {tuple(scale.shape)} cannot broadcast to weight shape {tuple(weight.shape)}"
        ) from e


def load_dequantized(
    f,
    key: str,
    qmeta: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, str | None]:
    t = f.get_tensor(key)
    fmt = quant_format_for_key(qmeta, key)
    t = reinterpret_quant_storage(t, fmt, key)
    t = t.to(device=device)

    # Convert first; arithmetic directly on FP8 is intentionally avoided.
    if not (t.is_floating_point() or t.dtype in {torch.float8_e4m3fn, torch.float8_e5m2}):
        raise TypeError(f"{key}: unsupported tensor dtype {t.dtype}")
    out = t.to(torch.float32)
    del t

    scale_key = find_scale_key(f, key)
    if scale_key is not None:
        scale = f.get_tensor(scale_key)
        scale = broadcast_scale(scale, out, scale_key)
        out.mul_(scale)
        del scale
    return out, scale_key


def tensor_fro_norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x).item())


def choose_rank_from_energy(s: torch.Tensor, total_energy: float, target: float, min_rank: int, cap: int) -> tuple[int, float]:
    if total_energy <= 0.0:
        return 0, 1.0
    vals = (s[:cap].double() ** 2).cumsum(0) / total_energy
    idx = torch.nonzero(vals >= target)
    if idx.numel() == 0:
        r = cap
    else:
        r = int(idx[0].item()) + 1
    r = max(min_rank, min(r, cap))
    captured = float(vals[r - 1].item()) if r > 0 else 1.0
    return r, captured


def low_rank_svd(
    delta: torch.Tensor,
    max_rank: int,
    energy_target: float | None,
    min_rank: int,
    method: str,
    oversample: int,
    niter: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float]:
    if delta.ndim != 2:
        raise ValueError("low_rank_svd expects a 2D tensor")
    m, n = delta.shape
    min_dim = min(m, n)
    cap = min(max_rank, min_dim)
    if cap < 1:
        raise ValueError(f"invalid rank cap for shape {tuple(delta.shape)}")

    # Avoid materializing a float64 copy of a giant Krea2 matrix.
    total_norm = float(torch.linalg.vector_norm(delta).item())
    total_energy = total_norm * total_norm
    if total_energy == 0.0:
        empty_u = delta.new_zeros((m, 0))
        empty_s = delta.new_zeros((0,))
        empty_vh = delta.new_zeros((0, n))
        return empty_u, empty_s, empty_vh, 0, 1.0

    use_exact = method == "exact"
    if method == "auto":
        # Exact SVD is reasonable if we are already asking for almost the full basis.
        use_exact = min_dim <= max(256, cap + oversample)

    if use_exact:
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    else:
        q = min(min_dim, cap + max(0, oversample))
        # svd_lowrank returns V (n x q), not Vh.
        U, S, V = torch.svd_lowrank(delta, q=q, niter=niter)
        order = torch.argsort(S, descending=True)
        U = U[:, order]
        S = S[order]
        V = V[:, order]
        Vh = V.transpose(0, 1)

    if energy_target is None:
        r = cap
        captured = float((S[:r].double() ** 2).sum().item() / total_energy)
    else:
        r, captured = choose_rank_from_energy(S, total_energy, energy_target, min_rank, cap)

    return U[:, :r], S[:r], Vh[:r, :], r, min(captured, 1.0)


def make_lora_factors(
    U: torch.Tensor,
    S: torch.Tensor,
    Vh: torch.Tensor,
    balance: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return A [r,in], B [out,r] such that B@A ~= delta."""
    if balance == "sqrt":
        root = torch.sqrt(S)
        A = root[:, None] * Vh
        B = U * root[None, :]
    else:
        A = Vh
        B = U * S[None, :]
    return A, B


def store_lora(
    out: dict[str, torch.Tensor],
    target_base: str,
    A: torch.Tensor,
    B: torch.Tensor,
    rank: int,
    fmt: str,
    dtype: torch.dtype,
) -> None:
    A = A.to(device="cpu", dtype=dtype).contiguous()
    B = B.to(device="cpu", dtype=dtype).contiguous()
    alpha = torch.tensor(float(rank), dtype=torch.float32)

    if fmt == "comfy-ab":
        out[f"{target_base}.lora_A.weight"] = A
        out[f"{target_base}.lora_B.weight"] = B
    else:
        out[f"{target_base}.lora_down.weight"] = A
        out[f"{target_base}.lora_up.weight"] = B
    # alpha/rank == 1.0, preserving B@A exactly at LoRA strength 1.
    out[f"{target_base}.alpha"] = alpha


def main() -> int:
    args = parse_args()

    if args.rank < 1:
        raise SystemExit("--rank must be >= 1")
    if args.min_rank < 1 or args.min_rank > args.rank:
        raise SystemExit("--min-rank must be >= 1 and <= --rank")
    if args.energy is not None and not (0.0 < args.energy <= 1.0):
        raise SystemExit("--energy must be in (0, 1]")
    if args.oversample < 0 or args.niter < 0:
        raise SystemExit("--oversample and --niter must be >= 0")
    if not (0.0 <= args.warn_energy <= 1.0):
        raise SystemExit("--warn-energy must be in [0, 1]")
    if args.warn_relative_delta < 0.0:
        raise SystemExit("--warn-relative-delta must be >= 0")

    path_a = Path(args.a)
    path_b = Path(args.b)
    out_path = Path(args.output)
    if not path_a.is_file():
        raise SystemExit(f"A not found: {path_a}")
    if not path_b.is_file():
        raise SystemExit(f"B not found: {path_b}")
    if path_a.resolve() == path_b.resolve():
        raise SystemExit("A and B are the same file")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    include_re = [re.compile(x) for x in args.include]
    exclude_re = [re.compile(x) for x in args.exclude]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    odtype = output_dtype(args.output_dtype)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"A       : {path_a}")
    print(f"B/base  : {path_b}")
    print(f"Output  : {out_path}")
    print(f"Device  : {device}")
    print(f"Rank    : <= {args.rank}" + (f", energy >= {args.energy:.4f}" if args.energy else " (fixed)"))
    print(f"Format  : {args.format}, {args.output_dtype}")
    print(f"Targets : {args.target_set}")
    print("Direction: delta = A - B; apply LoRA to B at strength 1.0 to approximate A")
    print()

    output_sd: dict[str, torch.Tensor] = {}
    reports: list[LayerReport] = []
    skipped: list[dict[str, Any]] = []
    started = time.time()
    global_delta_energy = 0.0
    global_kept_energy = 0.0
    low_energy_layers = 0
    large_delta_layers = 0

    with safe_open(str(path_a), framework="pt", device="cpu") as fa, safe_open(str(path_b), framework="pt", device="cpu") as fb:
        keys_a = set(fa.keys())
        keys_b = set(fb.keys())
        qmeta_a = parse_quant_metadata(fa)
        qmeta_b = parse_quant_metadata(fb)

        data_keys_a = {k for k in keys_a if not is_aux_key(k)}
        data_keys_b = {k for k in keys_b if not is_aux_key(k)}
        only_a = sorted(data_keys_a - data_keys_b)
        only_b = sorted(data_keys_b - data_keys_a)
        if only_a or only_b:
            msg = f"parameter key mismatch: only A={len(only_a)}, only B={len(only_b)}"
            if args.strict_keys:
                raise RuntimeError(msg + f"\nonly A sample={only_a[:8]}\nonly B sample={only_b[:8]}")
            print(f"WARNING: {msg}; unmatched keys will be skipped")

        common = sorted(data_keys_a & data_keys_b)
        preselected = [k for k in common if passes_filters(k, include_re, exclude_re)]
        if args.target_set == "krea2":
            selected = [k for k in preselected if krea2_canonical_lora_base(k) is not None]
            rejected = len(preselected) - len(selected)
            print(
                f"Common parameter keys: {len(common)}; filter matches: {len(preselected)}; "
                f"canonical Krea2 LoRA targets: {len(selected)}; rejected non-LoRA params: {rejected}"
            )
        else:
            selected = preselected
            print(f"Common parameter keys: {len(common)}; selected by filters: {len(selected)}")

        layer_index = 0
        for key in selected:
            shape_a = shape_of(fa, key)
            shape_b = shape_of(fb, key)
            if shape_a != shape_b:
                raise RuntimeError(f"shape mismatch for {key}: A={shape_a}, B={shape_b}")

            ndim = len(shape_a)
            is_matrix = ndim == 2
            if not is_matrix and not args.include_nonmatrix_diff:
                skipped.append({"key": key, "shape": list(shape_a), "reason": "non-2D parameter"})
                continue

            layer_index += 1
            prefix = f"[{layer_index:03d}]"
            if args.dry_run:
                sa = find_scale_key(fa, key)
                sb = find_scale_key(fb, key)
                print(f"{prefix} {key} shape={shape_a} matrix={is_matrix} scaleA={sa} scaleB={sb}")
                continue

            a, scale_a = load_dequantized(fa, key, qmeta_a, device)
            b, scale_b = load_dequantized(fb, key, qmeta_b, device)
            if a.shape != b.shape:
                raise RuntimeError(f"dequantized shape mismatch for {key}: A={tuple(a.shape)}, B={tuple(b.shape)}")

            # In-place subtraction saves one full matrix allocation.
            a.sub_(b)
            delta = a
            b_norm = tensor_fro_norm(b)
            del b

            delta_norm = tensor_fro_norm(delta)
            # ||A|| can be recovered as ||B + delta||, but that creates another large tensor.
            # B norm is sufficient as a stable relative scale for diagnostics.
            rel = delta_norm / max(b_norm, 1e-30)
            layer_energy = delta_norm * delta_norm
            global_delta_energy += layer_energy
            if rel > args.warn_relative_delta:
                large_delta_layers += 1
                print(
                    f"{prefix} WARNING unusually large delta: ||A-B||/||B||={rel:.6g} > "
                    f"{args.warn_relative_delta:.6g} for {key}"
                )

            if is_matrix:
                if delta_norm <= args.min_delta_norm:
                    print(f"{prefix} SKIP {key} shape={shape_a} delta_norm={delta_norm:.6g}")
                    skipped.append({"key": key, "shape": list(shape_a), "reason": "delta norm threshold"})
                    del delta
                    continue

                t0 = time.time()
                U, S, Vh, rank, captured = low_rank_svd(
                    delta,
                    max_rank=args.rank,
                    energy_target=args.energy,
                    min_rank=args.min_rank,
                    method=args.method,
                    oversample=args.oversample,
                    niter=args.niter,
                )
                if rank == 0:
                    skipped.append({"key": key, "shape": list(shape_a), "reason": "zero delta"})
                    del delta, U, S, Vh
                    continue

                A_factor, B_factor = make_lora_factors(U, S, Vh, args.factor_balance)
                if args.target_set == "krea2":
                    target = krea2_canonical_lora_base(key)
                    if target is None:
                        raise RuntimeError(f"internal error: selected non-Krea2 target {key}")
                else:
                    target = normalize_target_base(key, args.prefix)
                store_lora(output_sd, target, A_factor, B_factor, rank, args.format, odtype)
                kept_energy = captured * layer_energy
                global_kept_energy += kept_energy
                elapsed = time.time() - t0
                warn = ""
                if captured < args.warn_energy:
                    low_energy_layers += 1
                    warn += (
                        f"  WARNING LOW-RANK FIT: {captured:.4f} < {args.warn_energy:.4f}; "
                        "this layer is not well represented at the requested rank"
                    )
                if args.energy is not None and captured + 1e-7 < args.energy:
                    warn = f"  WARNING target {args.energy:.4f} not reached at rank cap"
                print(
                    f"{prefix} {key} {shape_a} -> r={rank:<4d} "
                    f"energy={captured:.6f} rel_delta={rel:.5g} svd={elapsed:.2f}s{warn}"
                )
                reports.append(
                    LayerReport(
                        key=key,
                        shape=list(shape_a),
                        kind="lora",
                        rank=rank,
                        captured_energy=captured,
                        delta_fro_norm=delta_norm,
                        relative_delta_norm=rel,
                        scale_a=scale_a,
                        scale_b=scale_b,
                    )
                )
                del delta, U, S, Vh, A_factor, B_factor
            else:
                if delta_norm <= args.nonmatrix_threshold:
                    skipped.append({"key": key, "shape": list(shape_a), "reason": "non-matrix delta threshold"})
                    del delta
                    continue
                target = normalize_target_base(key, args.prefix)
                output_sd[f"{target}.diff"] = delta.to(device="cpu", dtype=odtype).contiguous()
                global_kept_energy += layer_energy
                print(f"{prefix} DIFF {key} {shape_a} norm={delta_norm:.6g} rel_delta={rel:.5g}")
                reports.append(
                    LayerReport(
                        key=key,
                        shape=list(shape_a),
                        kind="exact-diff",
                        rank=None,
                        captured_energy=1.0,
                        delta_fro_norm=delta_norm,
                        relative_delta_norm=rel,
                        scale_a=scale_a,
                        scale_b=scale_b,
                        note="hybrid non-matrix exact diff",
                    )
                )
                del delta

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if args.dry_run:
        print("\nDry run complete; no output written.")
        return 0

    if not output_sd:
        raise RuntimeError("No LoRA/diff tensors were produced. Check filters and input checkpoints.")

    meta = {
        "title": out_path.stem,
        "architecture": "Krea2",
        "conversion": "checkpoint_delta_to_lora",
        "delta_direction": "A-B",
        "base_checkpoint": path_b.name,
        "target_checkpoint": path_a.name,
        "rank_cap": str(args.rank),
        "energy_target": "" if args.energy is None else str(args.energy),
        "svd_method": args.method,
        "factor_balance": args.factor_balance,
        "lora_format": args.format,
        "target_set": args.target_set,
        "output_dtype": args.output_dtype,
        "created_by": "krea2_delta_to_lora.py",
    }
    save_file(output_sd, str(out_path), metadata=meta)

    elapsed_total = time.time() - started
    global_capture = (global_kept_energy / global_delta_energy) if global_delta_energy > 0.0 else 1.0
    global_rel_error = math.sqrt(max(0.0, 1.0 - min(global_capture, 1.0)))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {out_path} ({size_mb:.1f} MiB)")
    print(f"LoRA/exact-diff targets: {len(reports)}")
    print(f"Output tensors: {len(output_sd)}")
    print(f"Global captured delta energy: {global_capture:.6f}")
    print(f"Global relative low-rank residual (Frobenius): {global_rel_error:.6f}")
    print(f"Low-energy layers (< {args.warn_energy:.3f}): {low_energy_layers}")
    print(f"Large-delta layers (> {args.warn_relative_delta:.3f}): {large_delta_layers}")
    if low_energy_layers:
        print("WARNING: the A-B difference is not sufficiently low-rank at this rank; strength 1.0 may be destructive.")
    if large_delta_layers:
        print("WARNING: unusually large A-B deltas detected; verify FP8 scaling and that B is the exact base used at inference.")
    print(f"Elapsed: {elapsed_total:.1f}s")

    if not args.no_report:
        report_path = Path(args.report) if args.report else out_path.with_suffix(out_path.suffix + ".report.json")
        report = {
            "a": str(path_a),
            "b": str(path_b),
            "output": str(out_path),
            "direction": "A-B",
            "options": {
                "rank": args.rank,
                "energy": args.energy,
                "min_rank": args.min_rank,
                "method": args.method,
                "oversample": args.oversample,
                "niter": args.niter,
                "device": str(device),
                "output_dtype": args.output_dtype,
                "format": args.format,
                "prefix": args.prefix,
                "factor_balance": args.factor_balance,
                "include_nonmatrix_diff": args.include_nonmatrix_diff,
                "target_set": args.target_set,
                "warn_energy": args.warn_energy,
                "warn_relative_delta": args.warn_relative_delta,
            },
            "summary": {
                "global_captured_delta_energy": global_capture,
                "global_relative_lowrank_residual": global_rel_error,
                "low_energy_layers": low_energy_layers,
                "large_delta_layers": large_delta_layers,
            },
            "layers": [asdict(x) for x in reports],
            "skipped": skipped,
            "elapsed_seconds": elapsed_total,
            "output_size_bytes": out_path.stat().st_size,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report: {report_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
