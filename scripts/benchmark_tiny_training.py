#!/usr/bin/env python3
"""Bounded synthetic Krea2 training experiment (not a production speed claim).

Run this same script in separate processes with PYTHONPATH selecting the desired
source tree. It uses real reduced SingleStreamDiT, native adapters, checkpointed
backward and AdamW/EMA, but synthetic latents/conditioning and no pretrained files.
No downloads, real datasets, trackers, service changes or checkpoint overwrites.
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import logging
import math
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from krea2_trainer.krea2.krea2_mmdit import SingleMMDiTConfig, SingleStreamDiT
from krea2_trainer.krea2_train_network import Krea2NetworkTrainer, krea2_setup_parser
from krea2_trainer.krea2_post_training import Krea2PostTrainingTrainer
from krea2_trainer.krea2_flow_cpo import Krea2FlowCPOTrainer
from krea2_trainer.networks.lora_krea2 import create_arch_network
from krea2_trainer.training.flow_cpo import AdapterEMA
from krea2_trainer.training.parser_common import setup_parser_common
import krea2_trainer.training.trainer_base as trainer_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("tqd", "flow_dpo", "flow_cpo"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch", type=int, default=2, help="images for TQD, preference pairs for DPO/CPO")
    parser.add_argument("--seed", type=int, default=741)
    parser.add_argument("--collect-metrics", action="store_true", help="compute tracker metrics but do not contact a tracker")
    parser.add_argument("--profile", action="store_true", help="separate diagnostic run; never compare its timing to unprofiled runs")
    parser.add_argument("--output", type=Path, required=True, help="new output directory (must not already exist)")
    cli = parser.parse_args()
    if not 1 <= cli.steps <= 100 or not 1 <= cli.warmup <= 30 or not 1 <= cli.batch <= 4:
        parser.error("bounded fixture requires 1..100 steps, 1..30 warmup and 1..4 batch")
    cli.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    logging.disable(logging.INFO)
    device = torch.device("cuda:0" if cli.device == "cuda" else "cpu")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback for GPU evidence")
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(0.1, (4 * 1024**3) / total), device)
    torch.manual_seed(cli.seed)
    config = SingleMMDiTConfig(features=512, tdim=128, txtdim=256, heads=8, kvheads=2, multiplier=2,
                              layers=4, patch=2, channels=4, txtlayers=4, txtheads=4, txtkvheads=4)
    model = SingleStreamDiT(config).to(device=device, dtype=torch.bfloat16).requires_grad_(False)

    def attach(multiplier=1.0):
        network = create_arch_network(multiplier, 32, 16, None, None, model)
        network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
        return network.to(device=device, dtype=torch.float32)

    stage = None
    if cli.mode != "tqd":
        stage = attach().eval().requires_grad_(False)
        with torch.no_grad():
            for adapter in stage.unet_loras:
                adapter.lora_up.weight.normal_(std=.002)
    policy = attach()
    with torch.no_grad():
        for adapter in policy.unet_loras:
            adapter.lora_up.weight.normal_(std=.001)
    trainer = {"tqd": Krea2NetworkTrainer, "flow_dpo": Krea2PostTrainingTrainer, "flow_cpo": Krea2FlowCPOTrainer}[cli.mode]()
    if stage is not None:
        trainer.reference_network = stage
    if cli.mode == "flow_cpo":
        trainer.ema_network = attach(0.0)
        trainer.ema = AdapterEMA(policy, trainer.ema_network, .99)
    model.enable_gradient_checkpointing()
    model.train()
    arguments = ["--mixed_precision", "bf16", "--sdpa", "--gradient_checkpointing",
                 "--network_module", "krea2_trainer.networks.lora_krea2", "--network_dim", "32", "--network_alpha", "16",
                 "--optimizer_type", "AdamW", "--weighting_scheme", "none", "--learning_rate", "1e-4"]
    if cli.mode == "tqd":
        arguments += ["--timestep_sampling", "tqd_krea2_shift", "--tqd_quality_weighting"]
    else:
        arguments += ["--post_training", cli.mode, "--timestep_sampling", "uniform"]
        if cli.mode == "flow_dpo":
            arguments += ["--flow_dpo_beta", "1"]
    args = krea2_setup_parser(setup_parser_common()).parse_args(arguments)
    accelerator = SimpleNamespace(device=device, unwrap_model=lambda x: x,
                                  autocast=lambda: torch.autocast(device.type, dtype=torch.bfloat16),
                                  trackers=(object(),) if cli.collect_metrics else (),
                                  sync_gradients=True, optimizer_step_was_skipped=False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=.01)
    trainer.on_train_start(args, accelerator, policy, model, optimizer)
    profiler = None
    if cli.profile:
        from krea2_trainer.training.profiling import TrainingStepProfiler
        profiler = TrainingStepProfiler(cli.steps, cli.warmup, device, cli.output / "phases.json",
                                        metadata={"mode": cli.mode, "unit": "images" if cli.mode == "tqd" else "pairs"})
        trainer._step_profiler = profiler
    phase = getattr(trainer, "profile_phase", lambda name: nullcontext())
    torch.manual_seed(cli.seed + 1)
    latents = torch.randn(cli.batch, 4, 1, 32, 32, device=device)
    batch = dict(latents=latents, timesteps=None,
                 krea2_vl_embed=[torch.randn(64 + 16 * i, 4, 256, device=device) for i in range(cli.batch)])
    if cli.mode == "tqd":
        scores = tuple((.8, .3) for _ in range(cli.batch))
        batch.update(tqd_structure_score=torch.full((cli.batch,), .8, device=device),
                     tqd_detail_score=torch.full((cli.batch,), .3, device=device), tqd_score_values=scores)
    else:
        batch["rejected_latents"] = torch.randn_like(latents)
    immutable = [(module, {key: value.detach().cpu().clone() for key, value in module.state_dict().items()})
                 for module in ([model, stage] if stage is not None else [model])]
    initial_policy = torch.cat([p.detach().flatten().cpu() for p in policy.parameters()])
    evidence = {"initial_policy": initial_policy}
    times, losses = [], []
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    original_call = trainer.call_dit
    capture = True
    predictions = []

    def capture_call(*a, **kw):
        result = original_call(*a, **kw)
        if capture:
            predictions.append(result.pred.detach().clone())
        return result

    trainer.call_dit = capture_call
    torch.manual_seed(cli.seed + 2)
    for step in range(cli.warmup + cli.steps):
        if profiler is not None:
            profiler.begin_step(step, cli.batch)
        synchronize()
        if step == cli.warmup and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        noise = torch.randn_like(latents)
        with phase("process_batch"):
            loss, metrics = trainer.process_batch(args, accelerator, model, policy, batch, latents, noise, None,
                                                  torch.bfloat16, torch.float32, None, step)
        with phase("backward"):
            loss.backward()
        if step == 0:
            evidence["first_gradient"] = torch.cat([p.grad.detach().flatten().clone() for p in policy.parameters()])
        with phase("optimizer"):
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        with phase("post_optimizer_hook"):
            trainer.on_post_optimizer_step(args, accelerator, policy, model, True, step)
        synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        if profiler is not None:
            profiler.end_step(optimizer_updated=True)
        losses.append(loss.detach().item())
        if step >= cli.warmup:
            times.append(elapsed)
        if step == 0:
            for i, pred in enumerate(predictions):
                evidence[f"first_prediction_{i}"] = pred.cpu()
            evidence["first_rng_cpu"] = torch.get_rng_state()
            if device.type == "cuda":
                evidence["first_rng_cuda"] = torch.cuda.get_rng_state(device)
            capture = False
            predictions.clear()
    if profiler is not None:
        profiler.finish()
    for module, saved in immutable:
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value.cpu(), saved[key], rtol=0, atol=0)
        if any(p.requires_grad or p.grad is not None for p in module.parameters()):
            raise AssertionError("Frozen base/reference changed gradient state")
    final_policy = torch.cat([p.detach().flatten().cpu() for p in policy.parameters()])
    if torch.equal(initial_policy, final_policy) or not torch.isfinite(final_policy).all() or not all(map(math.isfinite, losses)):
        raise AssertionError("No real finite optimizer update")
    if cli.mode == "flow_cpo" and trainer.ema_updates != cli.steps + cli.warmup:
        raise AssertionError("Incorrect EMA update count")
    evidence["final_policy"] = final_policy
    if cli.mode == "flow_cpo":
        evidence["final_ema"] = torch.cat([p.detach().flatten().cpu() for p in trainer.ema_network.parameters()])
    save_file({key: value.cpu().contiguous() for key, value in evidence.items()}, str(cli.output / "comparison.safetensors"))
    policy.save_weights(str(cli.output / "tiny-policy.safetensors"), torch.float32, {"fixture": "synthetic-reduced-model-only"})
    ordered = sorted(times)
    result = {
        "scope": "synthetic reduced-model training experiment; not production-model or quality evidence",
        "source_module": trainer_module.__file__, "torch": torch.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "config": asdict(config), "mode": cli.mode, "units_per_step": cli.batch,
        "unit": "images" if cli.mode == "tqd" else "pairs", "rank": 32, "alpha": 16,
        "seed": cli.seed, "warmup_steps": cli.warmup, "measured_steps": cli.steps,
        "optimizer_updates": cli.steps + cli.warmup, "gradient_checkpointing": True,
        "metrics_collected_without_external_tracker": cli.collect_metrics, "profiling_enabled": cli.profile,
        "step_wall_ms": times, "median_ms": statistics.median(times),
        "p95_ms": ordered[math.ceil(.95 * len(ordered)) - 1],
        "units_per_second": cli.batch * len(times) / (sum(times) / 1000), "losses": losses,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "frozen_base_reference_unchanged": True, "policy_changed_and_finite": True,
        "ema_updates": trainer.ema_updates if cli.mode == "flow_cpo" else None,
    }
    (cli.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("mode", "median_ms", "p95_ms", "peak_allocated_bytes", "optimizer_updates")}, indent=2))


if __name__ == "__main__":
    main()
