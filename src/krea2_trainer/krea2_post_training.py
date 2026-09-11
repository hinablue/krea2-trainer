"""Krea2 RFT and FlowDPO with a frozen, explicitly identified reference.

Only the new incremental adapter is optimized/saved. A stage-one adapter remains
an independent frozen forward hook, not merged into scaled-FP8 base tensors.
"""

import copy
import json
import logging
from pathlib import Path

from safetensors import SafetensorError
from safetensors.torch import load_file
import torch

from krea2_trainer.dataset import config_utils
from krea2_trainer.krea2_train_network import Krea2NetworkTrainer
from krea2_trainer.networks import lora_krea2
from krea2_trainer.training.accelerator_setup import collator_class
from krea2_trainer.training.preference import (
    POST_TRAINING_STATE,
    build_reference_contract,
    flow_dpo_loss,
    per_image_mse,
    reference_adapter_context,
    validate_post_training_args,
    validate_resume_contract,
)

logger = logging.getLogger(__name__)


def _load_native_reference(path, transformer, option, *, multiplier=1.0, for_inference=False):
    """Preflight every tensor/target without modifying weights or forward hooks.

    Native Krea2 adapters may select a subset of Linear targets, but every
    supplied target must have a complete alpha/down/up triplet and be consumed.
    Do not use apply_to/load_state_dict as a validator: apply_to mutates the DiT.
    """
    try:
        weights = load_file(path, device="cpu")
        if not weights:
            raise ValueError("empty adapter")
        groups = {}
        suffixes = {"alpha", "lora_down.weight", "lora_up.weight"}
        for key, value in weights.items():
            name, _, suffix = key.partition(".")
            if not name.startswith("lora_unet_") or suffix not in suffixes:
                raise ValueError(f"unexpected key: {key}")
            if not value.numel() or value.is_complex() or not torch.isfinite(value).all():
                raise ValueError(f"empty or non-finite tensor: {key}")
            groups.setdefault(name, {})[suffix] = value
        for name, state in groups.items():
            if set(state) != suffixes:
                raise ValueError(f"incomplete alpha/down/up triplet: {name}")
            down, up, alpha = (state[key] for key in ("lora_down.weight", "lora_up.weight", "alpha"))
            if alpha.ndim != 0:
                raise ValueError(f"alpha must be scalar: {name}")
            if (
                down.ndim != 2
                or up.ndim != 2
                or not down.is_floating_point()
                or not up.is_floating_point()
                or down.shape[0] != up.shape[1]
            ):
                raise ValueError(f"incompatible Linear LoRA pair: {name}")
        reference = lora_krea2.create_arch_network_from_weights(multiplier, weights, unet=transformer, for_inference=for_inference)
        if not reference.unet_loras:
            raise ValueError("no compatible Krea2 LoRA modules")
        consumed = set()
        for module in reference.unet_loras:
            for suffix, tensor in (
                ("alpha", module.alpha),
                ("lora_down.weight", module.lora_down.weight),
                ("lora_up.weight", module.lora_up.weight),
            ):
                key = f"{module.lora_name}.{suffix}"
                if weights[key].shape != tensor.shape:
                    raise ValueError(f"incompatible target shape: {key}")
                consumed.add(key)
        if consumed != set(weights):
            raise ValueError("adapter contains unmatched target keys")
    except (OSError, SafetensorError, KeyError, ValueError, RuntimeError, IndexError, AssertionError) as exc:
        raise ValueError(f"--{option} must be a finite, nonempty, complete, compatible native Krea2 LoRA: {path}: {exc}") from exc
    return reference, weights


def _reject_tqd_config(value):
    if isinstance(value, dict):
        if value.get("tqd_score_file") is not None:
            raise ValueError("Post-training datasets cannot configure tqd_score_file")
        for child in value.values():
            _reject_tqd_config(child)
    elif isinstance(value, list):
        for child in value:
            _reject_tqd_config(child)


class Krea2PostTrainingTrainer(Krea2NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.reference_network = None
        self.reference_contract = None

    def _validate_args_and_init(self, args):
        validate_post_training_args(args)
        if args.post_training == "none":
            raise ValueError("Krea2PostTrainingTrainer requires rft or flow_dpo")
        if not super()._validate_args_and_init(args):
            return False
        # Small adapter/manifest hashes, but not a second read of the full DiT.
        # Stat-based DiT identity is deliberately labelled as weaker than SHA256.
        self.reference_contract = build_reference_contract(args)
        if args.resume:
            validate_resume_contract(args.resume, self.reference_contract)
        logger.info(
            "Post-training %s; immutable reference: %s", args.post_training, json.dumps(self.reference_contract, sort_keys=True)
        )
        return True

    def _build_dataset(self, args):
        # Check before the ordinary loader resolves/loads TQD score files.
        _reject_tqd_config(config_utils.load_user_config(args.dataset_config))
        source_group, _, current_epoch = super()._build_dataset(args)
        # Lazy import keeps the mathematical/trainer tests independent of data.
        from krea2_trainer.dataset.preference_dataset import build_preference_dataset_group

        group = build_preference_dataset_group(
            source_group,
            args.preference_manifest,
            args.post_training,
            batch_size=args.preference_batch_size,
            seed=args.seed,
            shared_epoch=current_epoch,
        )
        collator = collator_class(current_epoch, group if args.max_data_loader_n_workers == 0 else None)
        return group, collator, current_epoch

    def _build_network(self, args, accelerator, transformer, vae, weight_dtype):
        network_args = args
        if args.base_weights:
            multipliers = args.base_weights_multiplier or []
            # Validate ALL sources before merging even the first one. Retain
            # these exact loaded tensors rather than validating then rereading.
            sources = [
                _load_native_reference(
                    path,
                    transformer,
                    "base_weights",
                    multiplier=multipliers[i] if i < len(multipliers) else 1.0,
                    for_inference=True,
                )
                for i, path in enumerate(args.base_weights)
            ]
            for path, (reference, weights) in zip(args.base_weights, sources):
                reference.merge_to(None, transformer, weights, weight_dtype, "cpu")
                logger.info("Merged validated reference: %s, %d adapters", path, len(reference.unet_loras))
            # Skip the parent's permissive merge without patching it or changing
            # the caller's arguments/provenance. No forward hooks were installed.
            network_args = copy.copy(args)
            network_args.base_weights = None
            network_args.base_weights_multiplier = None
        if args.reference_lora:
            reference, weights = _load_native_reference(args.reference_lora, transformer, "reference_lora")
            reference.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
            # The stock load_weights is non-strict; do not use it here.
            reference.load_state_dict(weights, strict=True)
            reference.to(device=accelerator.device, dtype=torch.float32)
            reference.requires_grad_(False)
            reference.eval()
            self.reference_network = reference
            logger.info(
                "Frozen stage-one reference: %s, %d adapters, device=%s, dtype=float32",
                args.reference_lora,
                len(reference.unet_loras),
                accelerator.device,
            )
        # LoRAModule.apply_to captures the existing forward: this second stack
        # wraps base+reference, and starts with an exactly zero LoRA residual.
        network = super()._build_network(network_args, accelerator, transformer, vae, weight_dtype)
        if network is None or not network.unet_loras:
            raise ValueError("Post-training requires a nonempty incremental LoRA network")
        for module in network.unet_loras:
            if not torch.count_nonzero(module.lora_up.weight).eq(0):
                raise ValueError("The new post-training adapter must start with zero residual")
        self._freeze_reference()
        return network

    def _freeze_reference(self):
        # This stack is owned by the trainer, never a child of the policy/base,
        # so transformer.train() and policy.prepare_grad_etc cannot unfreeze it.
        if self.reference_network is not None:
            self.reference_network.requires_grad_(False)
            self.reference_network.eval()

    def on_train_start(self, args, accelerator, network, transformer, optimizer):
        self._freeze_reference()
        unwrapped = accelerator.unwrap_model(transformer)
        if any(parameter.requires_grad for parameter in unwrapped.parameters()):
            raise ValueError("Post-training base DiT parameters must remain frozen")
        if self.reference_network is not None:
            reference_ids = {id(parameter) for parameter in self.reference_network.parameters()}
            if any(id(parameter) in reference_ids for group in optimizer.param_groups for parameter in group["params"]):
                raise ValueError("Frozen stage-one reference must not be in the optimizer")

    def _register_hooks_and_resume(self, args, accelerator, network):
        if self.reference_contract is None:
            raise RuntimeError("Reference contract must be captured from original sources before model loading")
        contract = self.reference_contract

        def save_reference_contract(models, weights, output_dir):
            if accelerator.is_main_process:
                target = Path(output_dir) / POST_TRAINING_STATE
                temporary = target.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                temporary.replace(target)

        def load_reference_contract(models, input_dir):
            validate_resume_contract(input_dir, contract)

        # Register first so incompatible state fails BEFORE model/optimizer loads.
        accelerator.register_save_state_pre_hook(save_reference_contract)
        accelerator.register_load_state_pre_hook(load_reference_contract)
        super()._register_hooks_and_resume(args, accelerator, network)

    def extra_metadata(self, args):
        if self.reference_contract is None:
            raise RuntimeError("Missing post-training reference provenance")
        metadata = {
            "ss_post_training": args.post_training,
            "ss_preference_manifest": self.reference_contract["preference_manifest"]["path"],
            "ss_post_training_adapter": "incremental-on-identical-dit-plus-stage1",
            "ss_reference_contract": json.dumps(self.reference_contract, sort_keys=True),
        }
        if args.post_training == "flow_dpo":
            metadata["ss_flow_dpo_beta"] = str(args.flow_dpo_beta)
            metadata["ss_flow_dpo_objective"] = "mean-fp32-velocity-mse-beta-over-2-uniform"
        return metadata

    def _prepare_paired_batch(self, args, accelerator, batch, latents, noise, dit_dtype, network_dtype):
        """Shared strict FlowDPO/FlowCPO input convention; no objective logic."""
        method = "FlowCPO" if args.post_training == "flow_cpo" else "FlowDPO"
        if batch.get("timesteps") is not None:
            raise ValueError(f"{method} requires freshly sampled uniform timesteps, not dataset timestep buckets")
        if any(key.startswith("tqd_") for key in batch):
            raise ValueError(f"{method} batches cannot contain TQD scores or weights")
        rejected = batch.get("rejected_latents")
        if not isinstance(latents, torch.Tensor) or not isinstance(rejected, torch.Tensor) or latents.shape != rejected.shape:
            raise ValueError(f"{method} chosen and rejected latents must have identical shapes")
        if latents.ndim != 5 or latents.shape[2] != 1 or latents.shape[0] == 0 or latents.numel() == 0:
            raise ValueError(f"{method} requires nonempty single-frame latents [pairs, C, 1, H, W]")
        if not isinstance(noise, torch.Tensor) or noise.shape != latents.shape:
            raise ValueError(f"{method} shared noise must match chosen latents")
        pairs = latents.shape[0]
        embeds = batch.get("krea2_vl_embed")
        if isinstance(embeds, torch.Tensor):
            if embeds.ndim != 4 or any(size <= 0 for size in embeds.shape):
                raise ValueError(f"{method} dense krea2_vl_embed must be nonempty [pairs, tokens, layers, hidden]")
            # Normalize dense native caches without changing varlen list semantics.
            embeds = list(embeds.unbind(0))
        if not isinstance(embeds, (list, tuple)) or len(embeds) != pairs:
            raise ValueError(f"{method} requires one shared krea2_vl_embed per preference pair")
        if any(
            not isinstance(embed, torch.Tensor) or embed.ndim != 3 or any(size <= 0 for size in embed.shape) for embed in embeds
        ):
            raise ValueError(f"{method} krea2_vl_embed items must be nonempty tensors [tokens, layers, hidden]")
        if args.post_training == "flow_cpo":
            for value in (latents, rejected, noise, *embeds):
                if not value.is_floating_point():
                    raise ValueError("FlowCPO batch tensors must be floating point")
                if not torch.isfinite(value).all():
                    raise FloatingPointError("FlowCPO batch tensors must be finite")
        device = accelerator.device
        chosen = latents.to(device=device, dtype=network_dtype)
        rejected = self.scale_shift_latents(rejected).to(device=device, dtype=network_dtype)
        combined = torch.cat((chosen, rejected))
        noise = noise.to(device=device, dtype=network_dtype)
        paired_noise = torch.cat((noise, noise))
        t = torch.rand(pairs, device=device, dtype=torch.float32)
        paired_t = torch.cat((t, t))
        broadcast_t = paired_t.view(-1, 1, 1, 1, 1)
        noisy = ((1.0 - broadcast_t) * combined + broadcast_t * paired_noise).to(dtype=dit_dtype)
        # call_dit divides model time by 1000; interpolation uses unshifted t.
        timesteps = 1000.0 * paired_t + 1.0
        # call_dit re-reads batch['latents'], not only its positional argument.
        pair_batch = dict(batch, latents=combined, krea2_vl_embed=list(embeds) + list(embeds), timesteps=None)
        return pairs, t, combined, pair_batch, paired_noise, noisy, timesteps

    def process_batch(
        self,
        args,
        accelerator,
        transformer,
        network,
        batch,
        latents,
        noise,
        noise_scheduler,
        dit_dtype,
        network_dtype,
        vae,
        global_step,
    ):
        self._freeze_reference()
        if args.post_training == "rft":
            # Dataset adapter already selects/deduplicates winners; no rejected
            # branch, reward model, reference inference, or alternative RFT loss.
            return super().process_batch(
                args,
                accelerator,
                transformer,
                network,
                batch,
                latents,
                noise,
                noise_scheduler,
                dit_dtype,
                network_dtype,
                vae,
                global_step,
            )
        if args.post_training != "flow_dpo":
            raise ValueError("Unknown post-training method")
        pairs, t, combined, pair_batch, paired_noise, noisy, timesteps = self._prepare_paired_batch(
            args, accelerator, batch, latents, noise, dit_dtype, network_dtype
        )
        policy_network = accelerator.unwrap_model(network)
        base_model = accelerator.unwrap_model(transformer)
        with reference_adapter_context(policy_network, base_model):
            reference = self.call_dit(
                args,
                accelerator,
                transformer,
                combined,
                pair_batch,
                paired_noise,
                noisy,
                timesteps,
                network_dtype,
            )
            reference_mse = per_image_mse(reference.pred, reference.target)
        # Reference forward finishes and multiplier/train flags are restored
        # BEFORE this graph exists. Checkpoint recomputation sees policy forever.
        policy = self.call_dit(
            args,
            accelerator,
            transformer,
            combined,
            pair_batch,
            paired_noise,
            noisy,
            timesteps,
            network_dtype,
        )
        policy_mse = per_image_mse(policy.pred, policy.target)
        loss, metrics = flow_dpo_loss(
            policy_mse[:pairs],
            policy_mse[pairs:],
            reference_mse[:pairs],
            reference_mse[pairs:],
            args.flow_dpo_beta,
        )
        metrics["flow_dpo/timestep_mean"] = t.mean().item()
        return loss, metrics
