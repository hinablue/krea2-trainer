#!/usr/bin/env python3
"""
Krea2 BF16/FP16 -> ComfyUI-native NVFP4 converter.

The output stores eligible Linear weights as:
    <layer>.weight
    <layer>.weight_scale
    <layer>.weight_scale_2
    <layer>.comfy_quant

Activations are dynamically quantized by ComfyUI at inference time because this
converter intentionally does not write static input_scale tensors.

Recommended first run:
    python quant_krea2_nvfp4.py krea2_turbo_bf16.safetensors --dry-run

Convert:
    python quant_krea2_nvfp4.py \
        krea2_turbo_bf16.safetensors \
        krea2_turbo_nvfp4.safetensors \
        --profile krea2-safe \
        --device cuda \
        --verify \
        --verify-report krea2_nvfp4_report.tsv
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


TOOL_VERSION = "1.0.0"
NVFP4_LAYOUT = "TensorCoreNVFP4Layout"
NVFP4_FORMAT = "nvfp4"
NVFP4_BLOCK_SIZE = 16
NVFP4_PACKED_ALIGNMENT = 32

GENERIC_EXCLUDE_SEGMENT = re.compile(
    r"scale_shift|rope|rotary|rel_pos|pos_?embed|embedder|"
    r"gate_logits|router|routing|logit|temperature|"
    r"(?:^|_)time|temb|t_emb|guidance|register|adapter|"
    r"(?:^|_)(?:final|head|proj_out|out_layer)(?:_|$)"
)

KREA2_SAFE_EXCLUDE = re.compile(
    r"(?:^|\.)(?:first|last|tmlp|txtmlp|tproj)(?:\.|$)|"
    r"(?:^|\.)txtfusion\.(?:projector|refiner_blocks)(?:\.|$)"
)

MAIN_KREA2_BLOCK = re.compile(r"(?:^|\.)blocks\.\d+\.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a single-file Krea2 BF16/FP16 safetensors model to ComfyUI-native NVFP4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("src", type=Path, help="Source Krea2 .safetensors file.")
    parser.add_argument(
        "dst",
        type=Path,
        nargs="?",
        help="Destination .safetensors file. Derived automatically when omitted.",
    )
    parser.add_argument(
        "--profile",
        choices=("krea2-safe", "main-blocks-only", "aggressive"),
        default="krea2-safe",
        help=(
            "krea2-safe quantizes indexed transformer blocks but preserves sensitive Krea2 paths; "
            "main-blocks-only quantizes only blocks.N.*; aggressive quantizes every eligible 2D weight "
            "except the generic denylist."
        ),
    )
    parser.add_argument(
        "--min-gemm",
        type=int,
        default=256,
        metavar="N",
        help="Skip a weight when min(out_features, in_features) is below N. Use 0 to disable.",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        metavar="REGEX",
        help="Force matching layer base names to remain in source precision.",
    )
    parser.add_argument(
        "--include",
        default=None,
        metavar="REGEX",
        help=(
            "Force matching weights to NVFP4 when their shape is technically eligible. "
            "This overrides built-in profile exclusions, but --exclude still wins."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device used one layer at a time for quantization.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cuda", "triton", "eager"),
        default="auto",
        help="Optional comfy-kitchen backend override.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the quantization plan without loading tensor data or writing output.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Dequantize each converted layer and calculate sampled cosine/relative error.",
    )
    parser.add_argument(
        "--verify-report",
        type=Path,
        default=None,
        help="Write per-layer verification metrics as TSV. Implies --verify.",
    )
    parser.add_argument(
        "--verify-samples",
        type=int,
        default=2_000_000,
        metavar="N",
        help="Maximum evenly-spaced elements used per layer for verification metrics.",
    )
    parser.add_argument(
        "--warn-thresh",
        type=float,
        default=10.0,
        metavar="PCT",
        help="Warn when sampled relative error exceeds this percentage.",
    )
    parser.add_argument(
        "--downcast-fp32",
        action="store_true",
        help="Downcast unquantized 2D FP32 weights to the dominant source compute dtype.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        metavar="N",
        help="Print progress after every N quantized layers.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing destination file.",
    )
    parser.add_argument(
        "--sha256",
        action="store_true",
        help="Calculate SHA256 after writing the output.",
    )
    parser.add_argument(
        "--allow-non-blackwell",
        action="store_true",
        help=(
            "Allow CUDA conversion on a GPU below SM 10.0. The file can be created, "
            "but native NVFP4 inference acceleration requires Blackwell."
        ),
    )
    return parser.parse_args()


def derive_destination(src: Path) -> Path:
    stem = src.stem
    replaced = re.sub(r"(?i)(bf16|fp16|fp32)", "nvfp4", stem)
    if replaced == stem:
        replaced = f"{stem}_nvfp4"
    return src.with_name(f"{replaced}.safetensors")


def roundup(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def shape_nvfp4_eligible(shape: tuple[int, ...]) -> tuple[bool, str]:
    if len(shape) != 2:
        return False, "not-2d"

    out_features, in_features = shape
    if out_features < 8:
        return False, "small-out-features"

    padded_in = roundup(in_features, NVFP4_BLOCK_SIZE)
    if padded_in % NVFP4_PACKED_ALIGNMENT != 0:
        return False, f"ineligible-input-alignment({in_features})"

    return True, "eligible"


def is_in_indexed_block(base: str) -> bool:
    segments = base.split(".")
    return any(segments[i].isdigit() for i in range(len(segments) - 1))


def has_generic_sensitive_segment(base: str) -> bool:
    return any(GENERIC_EXCLUDE_SEGMENT.search(segment) for segment in base.split("."))


def classify_layer(
    base: str,
    shape: tuple[int, ...],
    profile: str,
    min_gemm: int,
    include_re: re.Pattern[str] | None,
    exclude_re: re.Pattern[str] | None,
) -> tuple[bool, str]:
    eligible, reason = shape_nvfp4_eligible(shape)
    if not eligible:
        return False, reason

    forced_include = bool(include_re and include_re.search(base))

    if forced_include:
        quantize = True
        reason = "included-by-regex"
    elif profile == "main-blocks-only":
        quantize = bool(MAIN_KREA2_BLOCK.search(base)) and "txtfusion." not in base
        reason = "main-krea2-block" if quantize else "outside-main-krea2-blocks"
    elif profile == "krea2-safe":
        if not is_in_indexed_block(base):
            quantize = False
            reason = "not-in-indexed-block"
        elif KREA2_SAFE_EXCLUDE.search(base):
            quantize = False
            reason = "krea2-sensitive-path"
        elif has_generic_sensitive_segment(base):
            quantize = False
            reason = "generic-sensitive-path"
        else:
            quantize = True
            reason = "indexed-block"
    else:
        if has_generic_sensitive_segment(base):
            quantize = False
            reason = "generic-sensitive-path"
        else:
            quantize = True
            reason = "aggressive-eligible"

    if quantize and min_gemm > 0 and min(shape) < min_gemm:
        quantize = False
        reason = f"below-min-gemm({min(shape)})"

    if exclude_re and exclude_re.search(base):
        quantize = False
        reason = "excluded-by-regex"

    return quantize, reason


def comfy_quant_tensor() -> torch.Tensor:
    payload = json.dumps(
        {"format": NVFP4_FORMAT},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False")
    return torch.device(requested)


def dominant_compute_dtype(reader: Any, keys: list[str]) -> torch.dtype:
    counts: collections.Counter[str] = collections.Counter()
    for key in keys:
        if key.endswith(".weight"):
            counts[str(reader.get_slice(key).get_dtype())] += 1

    if counts.get("BF16", 0) >= counts.get("F16", 0) and counts.get("BF16", 0):
        return torch.bfloat16
    if counts.get("F16", 0):
        return torch.float16
    return torch.bfloat16


def normalize_pattern(base: str) -> str:
    return re.sub(r"\d+", "N", base)


def tensor_nbytes_from_shape(shape: tuple[int, ...], dtype_code: str) -> int:
    elements = 1
    for dim in shape:
        elements *= dim
    bytes_per_element = {
        "BF16": 2,
        "F16": 2,
        "F32": 4,
        "F64": 8,
        "I8": 1,
        "U8": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
    }.get(dtype_code, 2)
    return elements * bytes_per_element


def build_plan(
    reader: Any,
    keys: list[str],
    profile: str,
    min_gemm: int,
    include_re: re.Pattern[str] | None,
    exclude_re: re.Pattern[str] | None,
) -> tuple[list[tuple[str, tuple[int, int]]], collections.Counter[str], int, int]:
    plan: list[tuple[str, tuple[int, int]]] = []
    skipped: collections.Counter[str] = collections.Counter()
    quantized_params = 0
    source_tensor_bytes = 0

    for key in keys:
        slice_info = reader.get_slice(key)
        shape = tuple(slice_info.get_shape())
        dtype_code = str(slice_info.get_dtype())
        source_tensor_bytes += tensor_nbytes_from_shape(shape, dtype_code)

        if not key.endswith(".weight"):
            continue

        base = key[: -len(".weight")]
        quantize, reason = classify_layer(
            base=base,
            shape=shape,
            profile=profile,
            min_gemm=min_gemm,
            include_re=include_re,
            exclude_re=exclude_re,
        )
        if quantize:
            plan.append((base, (shape[0], shape[1])))
            quantized_params += shape[0] * shape[1]
        else:
            skipped[reason] += 1

    return plan, skipped, quantized_params, source_tensor_bytes


def print_plan(
    src: Path,
    dst: Path | None,
    profile: str,
    compute_dtype: torch.dtype,
    plan: list[tuple[str, tuple[int, int]]],
    skipped: collections.Counter[str],
    quantized_params: int,
    source_tensor_bytes: int,
) -> None:
    grouped: dict[str, list[Any]] = collections.defaultdict(lambda: [0, None])
    for base, shape in plan:
        pattern = normalize_pattern(base)
        grouped[pattern][0] += 1
        grouped[pattern][1] = shape

    estimated_nvfp4_bytes = int(quantized_params * (0.5 + 1.0 / NVFP4_BLOCK_SIZE))

    print(f"SRC: {src}")
    if dst is not None:
        print(f"DST: {dst}")
    print(f"profile: {profile}")
    print(f"passthrough/compute dtype: {compute_dtype}")
    print(f"source tensor bytes (header estimate): {source_tensor_bytes / 1024**3:.2f} GiB")

    print(f"\nQUANTIZE {len(plan)} weights as NVFP4:")
    for pattern in sorted(grouped):
        count, shape = grouped[pattern]
        print(f"  x{count:<4d} {str(shape):20s} {pattern}")

    print(
        f"\nquantized parameters: {quantized_params / 1e9:.3f}B"
        f"  estimated NVFP4 payload: {estimated_nvfp4_bytes / 1024**3:.2f} GiB"
    )

    print(f"\nLEAVE AS-IS ({sum(skipped.values())} weights):")
    for reason, count in skipped.most_common():
        print(f"  x{count:<4d} {reason}")


def validate_source(keys: list[str]) -> None:
    if any(key.endswith(".comfy_quant") for key in keys):
        raise RuntimeError(
            "Source already contains .comfy_quant tensors. Refusing to quantize an already quantized model."
        )
    if any(key.endswith(".weight_scale_2") for key in keys):
        raise RuntimeError(
            "Source contains NVFP4-style weight_scale_2 tensors. Use the original BF16/FP16 checkpoint."
        )


def inspect_runtime(device: torch.device, allow_non_blackwell: bool) -> tuple[Any, Any]:
    try:
        import comfy_kitchen as ck
        from comfy_kitchen.tensor import QuantizedTensor
    except ImportError as exc:
        raise RuntimeError(
            "comfy-kitchen is required. Install a version compatible with your existing PyTorch; "
            "do not replace the NVIDIA NGC torch build on DGX Spark."
        ) from exc

    print(f"torch: {torch.__version__}")
    print(f"torch CUDA runtime: {torch.version.cuda}")
    print(f"quantization device: {device}")
    print(f"comfy-kitchen backends: {ck.list_backends()}")

    if device.type == "cuda":
        capability = torch.cuda.get_device_capability(device)
        name = torch.cuda.get_device_name(device)
        print(f"GPU: {name}, capability={capability}")
        if capability < (10, 0) and not allow_non_blackwell:
            raise RuntimeError(
                f"{name} has SM {capability[0]}.{capability[1]}. "
                "Native NVFP4 Tensor Core inference requires Blackwell SM >= 10.0. "
                "Use --allow-non-blackwell only when intentionally creating the file for another machine."
            )

    return ck, QuantizedTensor


@torch.no_grad()
def quantize_weight(
    weight: torch.Tensor,
    device: torch.device,
    backend: str,
    ck: Any,
    quantized_tensor_cls: Any,
) -> tuple[Any, torch.Tensor]:
    weight_device = weight.contiguous().to(device=device, non_blocking=False)
    backend_context = contextlib.nullcontext() if backend == "auto" else ck.use_backend(backend)
    with backend_context:
        quantized = quantized_tensor_cls.from_float(weight_device, NVFP4_LAYOUT)
    return quantized, weight_device


@torch.no_grad()
def verification_metrics(
    quantized: Any,
    source_device: torch.Tensor,
    max_samples: int,
    backend: str,
    ck: Any,
) -> tuple[float, float]:
    backend_context = contextlib.nullcontext() if backend == "auto" else ck.use_backend(backend)
    with backend_context:
        reconstructed = quantized.dequantize()

    source_flat = source_device.reshape(-1)
    recon_flat = reconstructed.reshape(-1)
    count = source_flat.numel()

    if max_samples > 0 and count > max_samples:
        step = max(1, count // max_samples)
        source_flat = source_flat[::step][:max_samples]
        recon_flat = recon_flat[::step][:max_samples]

    source_float = source_flat.float()
    recon_float = recon_flat.float()
    cosine = torch.nn.functional.cosine_similarity(source_float, recon_float, dim=0).item()
    relative_error = (
        (recon_float - source_float).norm()
        / source_float.norm().clamp(min=1e-30)
    ).item() * 100.0
    del reconstructed, source_float, recon_float
    return cosine, relative_error


def cpu_state_dict(quantized: Any, prefix: str) -> dict[str, torch.Tensor]:
    return {
        key: tensor.detach().to("cpu").contiguous()
        for key, tensor in quantized.state_dict(prefix).items()
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_destination(dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {dst}. Pass --overwrite to replace it.")
    dst.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    args.src = args.src.expanduser().resolve()

    if args.src.suffix.lower() != ".safetensors":
        raise ValueError("This converter accepts a single .safetensors source file only.")
    if not args.src.is_file():
        raise FileNotFoundError(f"Source file not found: {args.src}")
    if args.min_gemm < 0:
        raise ValueError("--min-gemm must be >= 0")
    if args.verify_samples < 0:
        raise ValueError("--verify-samples must be >= 0")
    if args.verify_report is not None:
        args.verify = True

    if args.dst is None:
        args.dst = derive_destination(args.src)
    args.dst = args.dst.expanduser().resolve()

    if args.src == args.dst:
        raise ValueError("Source and destination must be different files.")

    include_re = re.compile(args.include) if args.include else None
    exclude_re = re.compile(args.exclude) if args.exclude else None

    with safe_open(str(args.src), framework="pt", device="cpu") as reader:
        source_metadata = dict(reader.metadata() or {})
        keys = list(reader.keys())
        validate_source(keys)
        compute_dtype = dominant_compute_dtype(reader, keys)
        plan, skipped, quantized_params, source_tensor_bytes = build_plan(
            reader, keys, args.profile, args.min_gemm, include_re, exclude_re
        )

        print_plan(
            args.src, args.dst, args.profile, compute_dtype,
            plan, skipped, quantized_params, source_tensor_bytes
        )

        if not plan:
            raise RuntimeError(
                "No weights were selected for NVFP4. Check tensor names and try --profile aggressive."
            )
        if args.dry_run:
            print("\n[dry-run] Nothing was written.")
            return 0

        ensure_destination(args.dst, args.overwrite)
        device = select_device(args.device)
        ck, QuantizedTensor = inspect_runtime(device, args.allow_non_blackwell)

        free_bytes = shutil.disk_usage(args.dst.parent).free
        source_file_bytes = args.src.stat().st_size
        if free_bytes < max(8 * 1024**3, source_file_bytes // 3):
            print(
                "WARNING: destination filesystem may have insufficient free space "
                f"({free_bytes / 1024**3:.2f} GiB available).",
                file=sys.stderr,
            )

        quantized_bases = {base for base, _ in plan}
        out: dict[str, torch.Tensor] = {}
        metrics: list[tuple[float, float, str, tuple[int, int]]] = []
        converted = 0
        start = time.time()

        for key in keys:
            tensor = reader.get_tensor(key)

            if not key.endswith(".weight"):
                out[key] = tensor
                continue

            base = key[: -len(".weight")]
            if base not in quantized_bases:
                if args.downcast_fp32 and tensor.dtype == torch.float32 and tensor.ndim == 2:
                    out[key] = tensor.to(compute_dtype)
                else:
                    out[key] = tensor
                continue

            quantized, source_device = quantize_weight(
                tensor, device, args.backend, ck, QuantizedTensor
            )

            if args.verify:
                cosine, relative_error = verification_metrics(
                    quantized, source_device, args.verify_samples, args.backend, ck
                )
                metrics.append(
                    (relative_error, cosine, base, (tensor.shape[0], tensor.shape[1]))
                )
                if relative_error > args.warn_thresh:
                    print(
                        f"  WARN high error: {base} "
                        f"relerr={relative_error:.3f}% cos={cosine:.6f}",
                        flush=True,
                    )

            out.update(cpu_state_dict(quantized, f"{base}.weight"))
            out[f"{base}.comfy_quant"] = comfy_quant_tensor()
            converted += 1

            del tensor, quantized, source_device
            if device.type == "cuda" and converted % 8 == 0:
                torch.cuda.empty_cache()

            if converted % max(1, args.progress_every) == 0:
                elapsed = time.time() - start
                print(
                    f"  converted {converted}/{len(plan)} NVFP4 weights "
                    f"({elapsed:.1f}s) — {base}",
                    flush=True,
                )

        output_metadata = {
            **source_metadata,
            "quantization": NVFP4_FORMAT,
            "quantization_layout": NVFP4_LAYOUT,
            "quantization_profile": args.profile,
            "quantization_tool": "quant_krea2_nvfp4.py",
            "quantization_tool_version": TOOL_VERSION,
            "quantization_source": args.src.name,
            "quantized_weight_count": str(converted),
        }

        temporary = args.dst.with_name(f".{args.dst.name}.tmp")
        if temporary.exists():
            temporary.unlink()
        try:
            save_file(out, str(temporary), metadata=output_metadata)
            os.replace(temporary, args.dst)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise

    elapsed = time.time() - start
    print(
        f"\nDONE: quantized {converted} weights, wrote {len(out)} tensors "
        f"in {elapsed:.1f}s -> {args.dst}"
    )
    print(f"output size: {args.dst.stat().st_size / 1024**3:.2f} GiB")

    if metrics:
        metrics.sort(reverse=True)
        errors = [entry[0] for entry in metrics]
        print("\n=== sampled NVFP4 reconstruction error ===")
        print(
            f"mean={sum(errors)/len(errors):.3f}% "
            f"min={min(errors):.3f}% max={max(errors):.3f}% "
            f"layers={len(errors)}"
        )
        print("worst 10 layers:")
        for relative_error, cosine, base, shape in metrics[:10]:
            print(
                f"  {relative_error:7.3f}% cos={cosine:.6f} "
                f"{str(shape):20s} {base}"
            )

    if args.verify_report is not None and metrics:
        report = args.verify_report.expanduser().resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("w", encoding="utf-8") as handle:
            handle.write("relerr_pct\tcosine\tout_features\tin_features\tlayer\n")
            for relative_error, cosine, base, shape in metrics:
                handle.write(
                    f"{relative_error:.6f}\t{cosine:.8f}\t"
                    f"{shape[0]}\t{shape[1]}\t{base}\n"
                )
        print(f"verification report: {report}")

    if args.sha256:
        print(f"sha256: {sha256_file(args.dst)}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
