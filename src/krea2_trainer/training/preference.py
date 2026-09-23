"""Source-aligned FlowDPO objective and post-training safety contracts.

FlowDPO: https://arxiv.org/html/2501.13918, Appendix C (uniform time).
The author T2I implementation uses per-image mean velocity MSE and beta / 2:
https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/scripts/train_sd3_dpo.py#L918-L931
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from krea2_trainer.training.metrics import materialize_metrics
from krea2_trainer.utils.tensor_checks import all_finite
import torch.nn.functional as F
import toml


POST_TRAINING_STATE = "krea2_post_training_state.json"
POST_DEFAULTS = {
    "preference_manifest": None,
    "preference_batch_size": 1,
    "flow_dpo_beta": 100.0,
    "flow_cpo_beta": 0.5,
    "flow_cpo_lambda": 1.0,
    "flow_cpo_ema_decay": 0.99,
    "reference_lora": None,
}


class _PostTrainingOption(argparse.Action):
    """Remember explicit default-valued options too, including config reparses."""

    def __call__(self, parser, namespace, values, option_string=None):
        supplied = set(getattr(namespace, "_post_training_explicit", ()))
        supplied.add(self.dest)
        namespace._post_training_explicit = supplied
        setattr(namespace, self.dest, values)


def _positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def _nonnegative_float(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return result


def _ema_decay(value):
    result = _nonnegative_float(value)
    if result >= 1:
        raise argparse.ArgumentTypeError("must be in [0, 1)")
    return result


def add_post_training_arguments(parser):
    group = parser.add_argument_group("Krea2 preference post-training")
    group.add_argument("--post_training", choices=("none", "rft", "flow_dpo", "flow_cpo"), default="none")
    group.add_argument(
        "--preference_manifest",
        action=_PostTrainingOption,
        help="JSONL preference pairs keyed by original image basenames; requires post-training.",
    )
    group.add_argument(
        "--preference_batch_size",
        type=_positive_int,
        default=1,
        action=_PostTrainingOption,
        help="Winners (RFT) or pairs (FlowDPO/FlowCPO) per device microbatch; default 1.",
    )
    group.add_argument(
        "--flow_dpo_beta",
        type=_positive_float,
        default=100.0,
        action=_PostTrainingOption,
        help="FlowDPO beta, default 100: logit = beta/2 * (rejected_delta - chosen_delta), "
        "using FP32 per-image mean velocity MSE, not a sum; no timestep/SNR weights.",
    )
    group.add_argument(
        "--flow_cpo_beta",
        type=_positive_float,
        default=0.5,
        action=_PostTrainingOption,
        help="FlowCPO velocity interpolation/extrapolation beta; finite > 0, default 0.5.",
    )
    group.add_argument(
        "--flow_cpo_lambda",
        type=_nonnegative_float,
        default=1.0,
        action=_PostTrainingOption,
        help="FlowCPO rejected-branch MSE coefficient; finite >= 0, default 1.",
    )
    group.add_argument(
        "--flow_cpo_ema_decay",
        type=_ema_decay,
        default=0.99,
        action=_PostTrainingOption,
        help="FlowCPO FP32 incremental-adapter EMA decay in [0, 1), default 0.99.",
    )
    group.add_argument(
        "--reference_lora",
        action=_PostTrainingOption,
        help="Frozen stage-one Krea2 safetensors adapter stacked before the new incremental adapter. "
        "Without this or --base_weights, --dit itself is the reference (possibly premerged).",
    )
    return parser


def _explicit_options(args):
    supplied = set(getattr(args, "_post_training_explicit", ()))
    # The common config loader assigns namespace attributes directly, bypassing Actions.
    if getattr(args, "config_file", None):
        path = Path(args.config_file)
        if path.suffix != ".toml":
            path = Path(str(path) + ".toml")
        config = toml.load(path)
        supplied.update(config)
        for value in config.values():
            if isinstance(value, dict):
                supplied.update(value)
    return supplied


def validate_post_training_args(args):
    """Validate CLI/config before any dataset, encoder or checkpoint loading.

    Never rewrite a sampler or preset. Ordinary training is unchanged unless an
    otherwise ignored post-training-only option was explicitly supplied.
    """
    mode = getattr(args, "post_training", "none")
    method = "FlowCPO" if mode == "flow_cpo" else "FlowDPO"
    if mode not in ("none", "rft", "flow_dpo", "flow_cpo"):
        raise ValueError("--post_training must be none, rft, flow_dpo or flow_cpo")
    explicit = _explicit_options(args)
    if mode == "none":
        supplied = [key for key, default in POST_DEFAULTS.items() if key in explicit or getattr(args, key, default) != default]
        if supplied:
            raise ValueError(
                "Post-training-only options require --post_training rft, flow_dpo or flow_cpo: "
                + ", ".join("--" + key for key in supplied)
            )
        return
    if getattr(args, "mixed_precision", None) not in ("bf16", "fp16"):
        raise ValueError(
            "Post-training requires explicit --mixed_precision bf16 or fp16 (CLI or config); "
            "Accelerator environment defaults cannot identify the reference precision, and no autocast is unsupported."
        )
    if getattr(args, "preset", None):
        raise ValueError(
            "--preset is not supported for post-training: it overwrites explicit sampler/training options. "
            "Pass the desired settings explicitly (FlowDPO/FlowCPO require --timestep_sampling uniform)."
        )
    if not getattr(args, "preference_manifest", None):
        raise ValueError("--preference_manifest is required for post-training")
    batch_size = getattr(args, "preference_batch_size", 1)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("--preference_batch_size must be a positive integer")
    beta = getattr(args, "flow_dpo_beta", 100.0)
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta <= 0:
        raise ValueError("--flow_dpo_beta must be finite and positive")
    if mode != "flow_dpo" and ("flow_dpo_beta" in explicit or beta != 100.0):
        raise ValueError("--flow_dpo_beta is only meaningful with --post_training flow_dpo")
    for key in ("flow_cpo_beta", "flow_cpo_lambda", "flow_cpo_ema_decay"):
        value = getattr(args, key, POST_DEFAULTS[key])
        if mode != "flow_cpo":
            if key in explicit or value != POST_DEFAULTS[key]:
                raise ValueError(f"--{key} is only meaningful with --post_training flow_cpo")
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (key == "flow_cpo_beta" and value == 0)
            or (key == "flow_cpo_ema_decay" and value >= 1)
        ):
            raise ValueError(f"--{key} must be finite and in its supported range")
    if mode == "flow_cpo":
        # The hook precedes post-step max-norm mutation, and schedule-free train/eval
        # swaps can save a different policy than the one just used for EMA. Fail closed.
        if str(getattr(args, "optimizer_type", "AdamW")).lower() not in (
            "",
            "adamw",
            "adam",
            "sgd",
            "adamw8bit",
            "torch.optim.adamw",
            "torch.optim.adam",
            "torch.optim.sgd",
        ):
            raise ValueError("FlowCPO supports only AdamW, Adam, SGD or AdamW8bit; optimizer mode-swapping paths are unverified")
        for key in ("scale_weight_norms", "full_fp16", "full_bf16"):
            if getattr(args, key, None):
                raise ValueError(f"FlowCPO does not support --{key}: EMA requires the final FP32 policy matrices")
    if getattr(args, "network_module", None) not in (
        "krea2_trainer.networks.lora_krea2",
        "networks.lora_krea2",
    ):
        raise ValueError("Post-training requires --network_module krea2_trainer.networks.lora_krea2")
    forbidden = (
        "network_weights",
        "dim_from_weights",
        "no_metadata",
        "resume_from_huggingface",
        "cache_te_every_epoch",
        "turbo_dit",
        "turbo_dit_cache",
        "tqd_quality_weighting",
    )
    for key in forbidden:
        if getattr(args, key, None):
            raise ValueError(
                f"--{key} is not supported for post-training. Use --reference_lora for stage one "
                "and --resume for a compatible post-training state; metadata must remain enabled."
            )
    if getattr(args, "timestep_sampling", "") == "tqd_krea2_shift":
        raise ValueError("Post-training cannot use TQD timestep sampling")
    if getattr(args, "tqd_kappa_base", 2.0) != 2.0 or getattr(args, "tqd_kappa_max", 8.0) != 8.0:
        raise ValueError("Post-training cannot use TQD concentration options")
    if getattr(args, "num_timestep_buckets", None) is not None:
        raise ValueError("Post-training preference datasets do not support --num_timestep_buckets")
    if getattr(args, "base_weights", None):
        if getattr(args, "reference_lora", None):
            raise ValueError("--reference_lora cannot be combined with --base_weights")
        if getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False):
            raise ValueError("--base_weights merging is unsafe with FP8 post-training; use --reference_lora instead")
    if getattr(args, "base_weights_multiplier", None):
        if not getattr(args, "base_weights", None):
            raise ValueError("--base_weights_multiplier requires --base_weights")
        if len(args.base_weights_multiplier) > len(args.base_weights):
            raise ValueError("Too many --base_weights_multiplier values")
        if not all(math.isfinite(x) for x in args.base_weights_multiplier):
            raise ValueError("--base_weights_multiplier must be finite")
    # External reference adapters are not registered on the base transformer's
    # module tree; existing block-swap cannot place/stream this second stack.
    if mode in ("flow_dpo", "flow_cpo") or getattr(args, "reference_lora", None):
        if (
            str(getattr(args, "dynamo_backend", "NO")).upper() != "NO"
            or os.environ.get("ACCELERATE_DYNAMO_BACKEND", "NO").upper() != "NO"
        ):
            raise ValueError(f"{method}/stacked references require --dynamo_backend NO and no Accelerator Dynamo compilation")
        for key in ("compile", "blocks_to_swap", "block_swap_h2d_only", "gradient_checkpointing_cpu_offload"):
            if getattr(args, key, None):
                raise ValueError(f"--{key} is not yet supported with {method} or a stacked --reference_lora")
    if mode in ("flow_dpo", "flow_cpo"):
        if getattr(args, "timestep_sampling", None) != "uniform":
            raise ValueError(f"{method} requires explicit --timestep_sampling uniform; no automatic schedule override")
        if getattr(args, "weighting_scheme", None) != "none":
            raise ValueError(f"{method} requires --weighting_scheme none (no timestep/SNR weighting)")
        for key in ("min_timestep", "max_timestep", "num_timestep_buckets"):
            if getattr(args, key, None) is not None:
                raise ValueError(f"{method} does not support --{key}; uniform t must cover [0, 1)")
        if getattr(args, "preserve_distribution_shape", False):
            raise ValueError(f"{method} does not support --preserve_distribution_shape")
        # Reject altered controls which the uniform objective does not consume.
        for key, default in (
            ("discrete_flow_shift", 1.0),
            ("sigmoid_scale", 1.0),
            ("logit_mean", 0.0),
            ("logit_std", 1.0),
            ("mode_scale", 1.29),
        ):
            if getattr(args, key, default) != default:
                raise ValueError(f"{method} uniform sampling does not use --{key}")
        if getattr(args, "network_dropout", None) not in (None, 0, 0.0):
            raise ValueError(f"{method} requires deterministic adapters: --network_dropout must be zero")
        for item in getattr(args, "network_args", None) or ():
            key, separator, value = item.partition("=")
            if not separator:
                raise ValueError("--network_args must be key=value")
            if key in ("rank_dropout", "module_dropout", "neuron_dropout") and float(value) != 0:
                raise ValueError(f"{method} requires zero {key}")


def per_image_mse(prediction, target):
    """FP32 mean over non-batch dimensions, never sum over image pixels."""
    if prediction.shape != target.shape or prediction.ndim < 2 or prediction.shape[0] == 0 or prediction.numel() == 0:
        raise ValueError("Prediction and target must have identical nonempty [batch, ...] shapes")
    mse = (prediction.float() - target.float()).square().flatten(1).mean(1)
    if not torch.isfinite(mse).all():
        raise FloatingPointError("Non-finite per-image MSE in FlowDPO")
    return mse


def flow_dpo_loss(
    policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta, *, collect_metrics=True, metrics_as_tensors=False
):
    """Return (pair-mean loss, metrics) for four vectors of per-image MSE.

    logit = beta/2 * [(e_policy_l - stopgrad(e_ref_l))
                     - (e_policy_w - stopgrad(e_ref_w))].
    This constant-beta author-code convention has no extra (1-t)^2 factor.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("FlowDPO beta must be finite and positive")
    values = (policy_chosen, policy_rejected, reference_chosen, reference_rejected)
    if any(value.ndim != 1 or value.shape != policy_chosen.shape for value in values) or not policy_chosen.numel():
        raise ValueError("FlowDPO MSE inputs must have identical nonempty [pairs] shapes")
    if not all_finite(values):
        raise FloatingPointError("All four FlowDPO MSE branches must be finite")
    chosen = policy_chosen.float() - reference_chosen.detach().float()
    rejected = policy_rejected.float() - reference_rejected.detach().float()
    margin = rejected - chosen
    logits = (0.5 * beta) * margin
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite FlowDPO logit")
    loss = -F.logsigmoid(logits).mean()
    if not collect_metrics:
        return loss, {}
    metrics = {
        "flow_dpo/loss": loss.detach(),
        "flow_dpo/chosen_mse": policy_chosen.detach().float().mean(),
        "flow_dpo/rejected_mse": policy_rejected.detach().float().mean(),
        "flow_dpo/reference_chosen_mse": reference_chosen.detach().float().mean(),
        "flow_dpo/reference_rejected_mse": reference_rejected.detach().float().mean(),
        "flow_dpo/margin": margin.detach().mean(),
        "flow_dpo/win_rate": (margin.detach() > 0).float().mean(),
    }
    return loss, metrics if metrics_as_tensors else materialize_metrics(metrics)


@contextmanager
def reference_adapter_context(network, transformer):
    """Disable ONLY the incremental adapter; restore every flag on exceptions.

    Call before constructing the policy autograd graph. In particular, do not
    turn the adapter off between the policy forward and checkpointed backward.
    No parameter/buffer tensor is ever modified by this context manager.
    """
    modules = list(dict.fromkeys([*network.modules(), *transformer.modules()]))
    training = [(module, module.training) for module in modules]
    multipliers = [(module, module.multiplier) for module in network.modules() if hasattr(module, "multiplier")]
    if not multipliers or not getattr(network, "unet_loras", None):
        raise TypeError("Reference disabling requires the native Krea2 LoRA network")
    try:
        for module, _ in multipliers:
            module.multiplier = 0.0
        network.eval()
        transformer.eval()
        with torch.no_grad():
            yield
    finally:
        for module, multiplier in multipliers:
            module.multiplier = multiplier
        for module, was_training in training:
            module.training = was_training


def file_fingerprint(path, *, content_hash=False):
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"Checkpoint/manifest must be a local file: {source}")
    stat = source.stat()
    result = {
        "path": str(source),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "identity": "sha256+stat" if content_hash else "stat-only-not-content-hash",
    }
    if content_hash:
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = source.stat()
        if (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev) != (
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ino,
            stat.st_dev,
        ):
            raise ValueError(f"Source changed while fingerprinting: {source}")
        result["sha256"] = digest.hexdigest()
    return result


def build_reference_contract(args):
    """Record the immutable reference, not the current/resumed policy weights."""
    base_weights = getattr(args, "base_weights", None) or []
    multipliers = getattr(args, "base_weights_multiplier", None) or []
    contract = {
        "version": 1,
        "adapter_semantics": "incremental-on-identical-dit-plus-stage1",
        "post_training": args.post_training,
        "dit": file_fingerprint(args.dit),
        "reference_lora": file_fingerprint(args.reference_lora, content_hash=True) if args.reference_lora else None,
        "base_weights": [
            {"source": file_fingerprint(path, content_hash=True), "multiplier": multipliers[i] if i < len(multipliers) else 1.0}
            for i, path in enumerate(base_weights)
        ],
        "preference_manifest": file_fingerprint(args.preference_manifest, content_hash=True),
        "dataset_config": file_fingerprint(args.dataset_config, content_hash=True),
        "preference_batch_size": args.preference_batch_size,
        "flow_dpo_beta": args.flow_dpo_beta if args.post_training == "flow_dpo" else None,
        "objective": "mean-fp32-velocity-mse-beta-over-2-uniform" if args.post_training == "flow_dpo" else "native-flow-matching",
        "settings": {
            key: getattr(args, key, None)
            for key in (
                "network_module",
                "network_dim",
                "network_alpha",
                "network_args",
                "network_dropout",
                "fp8_base",
                "fp8_scaled",
                "mixed_precision",
                "timestep_sampling",
                "weighting_scheme",
                "discrete_flow_shift",
                "sigmoid_scale",
                "min_timestep",
                "max_timestep",
                "num_timestep_buckets",
                "preserve_distribution_shape",
                "logit_mean",
                "logit_std",
                "mode_scale",
                "sdpa",
                "flash_attn",
                "flash3",
                "xformers",
                "split_attn",
            )
        },
    }

    if args.post_training == "flow_cpo":
        # Do not change even the key set of the existing RFT/DPO version-1 contract.
        contract.pop("flow_dpo_beta")
        contract.update(
            objective="mean-fp32-mixed-velocity-mse-uniform-ema-lora",
            flow_cpo_beta=args.flow_cpo_beta,
            flow_cpo_lambda=args.flow_cpo_lambda,
            flow_cpo_ema_decay=args.flow_cpo_ema_decay,
            ema_semantics="fp32-second-stage-lora-matrices-after-successful-optimizer-update",
        )
        contract["settings"]["optimizer_type"] = args.optimizer_type or "AdamW"
    return contract


def validate_resume_contract(directory, expected):
    source = Path(directory) / POST_TRAINING_STATE
    try:
        saved = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Post-training resume requires valid {POST_TRAINING_STATE}; cannot use metadata-less state") from exc
    if saved != expected:
        raise ValueError(
            "Incompatible post-training resume reference/objective/data contract; "
            "use the identical original --dit, --reference_lora/--base_weights, manifest and training settings"
        )
