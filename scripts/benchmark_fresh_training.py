#!/usr/bin/env python3
"""Bounded fresh LoRA/TQD benchmark: real RAW weights, synthetic caches only.

Never resumes a LoRA or enters RFT/DPO/CPO. The output directory must be new.
Timing observes native loss-recorder boundaries (after the existing loss.item
synchronization); it does not add per-step CUDA synchronization. --profile is
an explicitly separate synchronized diagnostic, not an unprofiled A/B run.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from krea2_trainer.krea2_train_network import Krea2NetworkTrainer, main as train_main
from krea2_trainer.utils import train_utils
import krea2_trainer.training.trainer_base as trainer_module


def profile_window_memory(report, expected_steps):
    """Aggregate per-step CUDA peaks; the profiler resets them each step."""
    rows = report.get("steps", [])
    if (report.get("complete") is not True or report.get("collected_microsteps") != expected_steps
            or not rows or len(rows) != expected_steps):
        raise ValueError("Incomplete profiler window; cannot report a whole-window memory peak")
    result = {}
    for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
        values = [row.get(key) if isinstance(row, dict) else None for row in rows]
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("Missing or invalid per-step CUDA memory measurement")
        result[key] = max(values)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New isolated experiment directory")
    parser.add_argument("--variant", choices=("eager", "compiled"), default="compiled")
    parser.add_argument("--training-mode", choices=("standard", "tqd"), default="tqd")
    parser.add_argument("--steps", type=int, default=8, help="Measured steps, in addition to warmup")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--resolution", type=int, choices=(512, 1024), default=1024)
    parser.add_argument("--seed", type=int, default=741)
    parser.add_argument("--profile", action="store_true")
    cli = parser.parse_args()
    if not 2 <= cli.steps <= 24 or not 1 <= cli.warmup <= 12:
        parser.error("require 2..24 measured steps and 1..12 warmup steps")
    cli.dit = cli.dit.expanduser().resolve(strict=True)
    if not cli.dit.is_file():
        parser.error("--dit must be a local checkpoint file")
    if not torch.cuda.is_available():
        raise RuntimeError("This RAW benchmark requires CUDA; no silent CPU fallback")
    cli.output = cli.output.expanduser().absolute()
    cli.output.mkdir(parents=True, exist_ok=False)
    cache = cli.output / "cache"
    images = cli.output / "images"
    cache.mkdir()
    images.mkdir()
    torch.set_num_threads(1)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(.70, 80 * 1024**3 / total_memory), 0)
    checkpoint_stat = cli.dit.stat()
    torch.manual_seed(82)
    latent_side = cli.resolution // 8
    scores = []
    for i in range(4):
        name = f"fixture{i}"
        save_file({f"latents_1x{latent_side}x{latent_side}_bfloat16": torch.randn(16, 1, latent_side, latent_side,
                                                                                 dtype=torch.bfloat16)},
                  str(cache / f"{name}_{cli.resolution:04d}x{cli.resolution:04d}_kr2.safetensors"))
        save_file({"varlen_krea2_vl_embed_bfloat16": torch.randn(96, 12, 2560, dtype=torch.bfloat16)},
                  str(cache / f"{name}_kr2_te.safetensors"), metadata={"caption1": "synthetic benchmark only"})
        scores.append(dict(image_file=f"{name}.png", structure_score=.8, detail_score=.3))
    score_file = cli.output / "scores.jsonl"
    score_file.write_text("".join(json.dumps(row) + "\n" for row in scores))
    dataset = cli.output / "dataset.toml"
    # JSON-quoted paths are also valid TOML basic strings, including spaces.
    dataset.write_text(
        f"[general]\nresolution=[{cli.resolution},{cli.resolution}]\nbatch_size=1\nenable_bucket=false\n"
        f"[[datasets]]\nimage_directory={json.dumps(str(images))}\ncache_directory={json.dumps(str(cache))}\nnum_repeats=1\n"
        + (f"tqd_score_file={json.dumps(str(score_file))}\n" if cli.training_mode == "tqd" else "")
    )
    total_steps = cli.steps + cli.warmup
    arguments = [
        "--dit", str(cli.dit), "--dataset_config", str(dataset), "--sdpa", "--mixed_precision", "bf16",
        "--save_precision", "bf16", "--gradient_checkpointing", "--network_module", "krea2_trainer.networks.lora_krea2",
        "--network_dim", "32", "--network_alpha", "16", "--optimizer_type", "Adopt_adv",
        "--optimizer_args", "cautious_wd=true", "kourkoutas_beta=true", "use_atan2=true", "weight_decay=0.01",
        "--learning_rate", "5e-5", "--lr_scheduler", "constant_with_warmup", "--lr_warmup_steps", "2",
        "--timestep_sampling", "tqd_krea2_shift" if cli.training_mode == "tqd" else "krea2_shift",
        "--weighting_scheme", "none", "--max_train_steps", str(total_steps), "--max_data_loader_n_workers", "0",
        "--seed", str(cli.seed), "--output_dir", str(cli.output), "--output_name", "synthetic_benchmark",
        "--cuda_allow_tf32", "--cuda_cudnn_benchmark",
    ]
    if cli.training_mode == "tqd":
        arguments.append("--tqd_quality_weighting")
    if cli.variant == "compiled":
        arguments += ["--compile", "--compile_mode", "max-autotune-no-cudagraphs",
                      "--compile_dynamic", "auto", "--compile_cache_size_limit", "32"]
    if cli.profile:
        arguments += ["--profile_steps", str(cli.steps), "--profile_warmup_steps", str(cli.warmup)]

    moments, losses, frozen, final_memory = [], [], [], {}
    original_add = train_utils.LossRecorder.add
    original_start = Krea2NetworkTrainer.on_train_start

    def observed_add(recorder, epoch, step, loss):
        result = original_add(recorder, epoch=epoch, step=step, loss=loss)
        moments.append(time.perf_counter())
        losses.append(loss)
        if len(moments) == cli.warmup:
            torch.cuda.reset_peak_memory_stats()
        if len(moments) == total_steps:
            final_memory.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved())
        return result

    def checked_start(trainer, args, accelerator, network, transformer, optimizer):
        if args.post_training != "none" or args.max_train_steps != total_steps:
            raise AssertionError("This experiment must remain bounded fresh training")
        original_start(trainer, args, accelerator, network, transformer, optimizer)
        base = accelerator.unwrap_model(transformer)
        base_parameters = tuple(base.parameters())
        if any(parameter.requires_grad for parameter in base_parameters):
            raise AssertionError("Base parameters must be frozen")
        base_ids = {id(parameter) for parameter in base_parameters}
        if any(id(parameter) in base_ids for group in optimizer.param_groups for parameter in group["params"]):
            raise AssertionError("Optimizer must not own base parameters")
        frozen.extend((parameter, parameter._version) for parameter in base_parameters)

    train_utils.LossRecorder.add = observed_add
    Krea2NetworkTrainer.on_train_start = checked_start
    previous_argv = sys.argv
    started = time.perf_counter()
    try:
        sys.argv = ["krea2-train-lora", *arguments]
        train_main()
    finally:
        train_utils.LossRecorder.add = original_add
        Krea2NetworkTrainer.on_train_start = original_start
        sys.argv = previous_argv
    total_wall = time.perf_counter() - started
    if len(moments) != total_steps or not all(math.isfinite(value) for value in losses):
        raise AssertionError("Incomplete or nonfinite training run")
    if cli.profile:
        # Per-step profiling resets the allocator peak before every sample;
        # the final callback alone would report only the last sample's peak.
        report = json.loads((cli.output / "training-profile.json").read_text())
        final_memory = profile_window_memory(report, cli.steps)
    if any(parameter._version != version or parameter.requires_grad or parameter.grad is not None
           for parameter, version in frozen):
        raise AssertionError("Frozen base parameter state changed")
    after = cli.dit.stat()
    if (checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns, checkpoint_stat.st_ctime_ns) != (
        after.st_size, after.st_mtime_ns, after.st_ctime_ns
    ):
        raise AssertionError("RAW checkpoint file metadata changed during the run")
    checkpoint = cli.output / "synthetic_benchmark.safetensors"
    changed_up = False
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        if metadata["ss_steps"] != str(total_steps):
            raise AssertionError("Saved checkpoint step count differs from the requested cap")
        for key in handle.keys():
            value = handle.get_tensor(key)
            if not torch.isfinite(value).all():
                raise AssertionError("Nonfinite LoRA export")
            if "lora_up.weight" in key and torch.count_nonzero(value):
                changed_up = True
    if not changed_up:
        raise AssertionError("Native LoRA optimizer did not update zero-initialized up projections")
    times = [(moments[i] - moments[i - 1]) * 1000 for i in range(cli.warmup, total_steps)]
    ordered = sorted(times)
    result = {
        "scope": "real RAW fresh training with synthetic caches; not real-data convergence/quality evidence",
        "variant": cli.variant, "training_mode": cli.training_mode, "resolution": cli.resolution,
        "batch_size": 1, "rank": 32, "alpha": 16, "optimizer": "Adopt_adv", "seed": cli.seed,
        "source_module": trainer_module.__file__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "timing_boundary": "native LossRecorder.add after existing loss.item synchronization; consecutive complete steps",
        "profile_enabled": cli.profile, "warmup_steps": cli.warmup, "measured_steps": cli.steps,
        "optimizer_updates": total_steps, "step_wall_ms": times, "median_ms": statistics.median(times),
        "p95_ms": ordered[math.ceil(.95 * len(ordered)) - 1], "images_per_second": len(times) / (sum(times) / 1000),
        "total_wall_seconds_including_load_compile_save": total_wall, "losses": losses,
        "base_frozen_versions_unchanged": True, "base_excluded_from_optimizer": True,
        "native_lora_updated_and_finite": True, "raw_file_stat_unchanged": True,
        "raw_checkpoint": str(cli.dit), "raw_checkpoint_bytes": checkpoint_stat.st_size,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "compile_counters": {str(key): dict(value) for key, value in torch._dynamo.utils.counters.items()},
        **final_memory,
    }
    (cli.output / "benchmark.json").write_text(json.dumps(result, indent=2, default=str, allow_nan=False) + "\n")
    print("FRESH_BENCHMARK=" + json.dumps({key: result[key] for key in (
        "variant", "training_mode", "median_ms", "p95_ms", "peak_allocated_bytes", "optimizer_updates"
    )}))


if __name__ == "__main__":
    main()
