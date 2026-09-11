"""FlowCPO with a separate FP32 EMA of the incremental Krea2 LoRA.

Paper Eq. 13 / Algorithm 1: https://arxiv.org/html/2609.09905.
The immutable base and stage one are reused, never copied or optimizer-owned.
Only the policy is registered with Accelerator and exported as a native adapter.
"""

import copy
import logging

import torch

from krea2_trainer.krea2_post_training import Krea2PostTrainingTrainer
from krea2_trainer.networks import lora_krea2
from krea2_trainer.training.flow_cpo import AdapterEMA, flow_cpo_loss, old_adapter_context
from krea2_trainer.training.preference import validate_resume_contract

logger = logging.getLogger(__name__)


def _reject_optimizer_mode_swaps(optimizer):
    # AcceleratedOptimizer exposes train/eval itself even around ordinary AdamW.
    # Inspect the actual underlying optimizer, not that generic wrapper surface.
    while hasattr(optimizer, "optimizer"):
        optimizer = optimizer.optimizer
    if any(callable(getattr(optimizer, key, None)) for key in ("train", "eval")):
        raise ValueError("FlowCPO cannot use optimizer train/eval parameter swapping (including schedule-free optimizers)")


class Krea2FlowCPOTrainer(Krea2PostTrainingTrainer):
    """Public FlowCPO trainer; ``ema_network``, ``ema`` and ``ema_updates`` are inspectable."""

    def __init__(self):
        super().__init__()
        self.ema_network = None
        self.ema = None

    @property
    def ema_updates(self):
        return self.ema.updates if self.ema is not None else 0

    def _validate_args_and_init(self, args):
        if args.post_training != "flow_cpo":
            raise ValueError("Krea2FlowCPOTrainer requires --post_training flow_cpo")
        return super()._validate_args_and_init(args)

    def _build_network(self, args, accelerator, transformer, vae, weight_dtype):
        if self.ema is not None:
            raise RuntimeError("FlowCPO adapter chain may only be constructed once per trainer")
        # Parent installs immutable stage one, then native zero-residual policy.
        network = super()._build_network(args, accelerator, transformer, vae, weight_dtype)
        network.to(dtype=torch.float32)
        # Reconstruct ONLY the exact selected LoRA targets/ranks from native keys.
        # Attach last so EMA wraps policy -> stage one -> base. Never deepcopy DiT.
        # Discarded random initialization must not perturb the training RNG stream.
        with torch.random.fork_rng(devices=[]):
            ema_network = lora_krea2.create_arch_network_from_weights(
                0.0, network.state_dict(), unet=transformer, for_inference=False
            )
        ema_network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        ema_network.to(device=accelerator.device, dtype=torch.float32)
        self.ema = AdapterEMA(network, ema_network, args.flow_cpo_ema_decay)
        self.ema_network = ema_network
        logger.info(
            "FlowCPO EMA initialized before resume: %d native adapters, FP32, decay=%s", len(ema_network.unet_loras), self.ema.decay
        )
        return network

    def _prepare_with_accelerator(
        self,
        args,
        accelerator,
        transformer,
        network,
        optimizer,
        train_dataloader,
        lr_scheduler,
        weight_dtype,
        dit_dtype,
        dit_weight_dtype,
    ):
        self._require_ema()
        if self.ema_updates:
            raise RuntimeError("FlowCPO accelerator preparation must precede EMA updates and resume")
        prepared = super()._prepare_with_accelerator(
            args,
            accelerator,
            transformer,
            network,
            optimizer,
            train_dataloader,
            lr_scheduler,
            weight_dtype,
            dit_dtype,
            dit_weight_dtype,
        )
        # DDP preparation may broadcast rank-zero policy initialization. Recopy
        # that final initial policy BEFORE registering any resume hooks. This is
        # not on_train_start, and cannot overwrite an EMA loaded by resume.
        self.ema = AdapterEMA(accelerator.unwrap_model(prepared[1]), self.ema_network, args.flow_cpo_ema_decay)
        return prepared

    def get_optimizer(self, args, trainable_params):
        if not args.optimizer_type:
            # The shared parser documents AdamW as the empty-string default.
            args = copy.copy(args)
            args.optimizer_type = "AdamW"
        result = super().get_optimizer(args, trainable_params)
        _reject_optimizer_mode_swaps(result[2])
        return result

    def _require_ema(self):
        if self.ema is None or self.ema_network is None:
            raise RuntimeError("FlowCPO EMA must be initialized before resume or training")

    def _check_policy_mode(self, network):
        self._require_ema()
        for candidate, multiplier in ((network, 1.0), (self.ema_network, 0.0)):
            for module in candidate.modules():
                if hasattr(module, "multiplier") and (
                    not isinstance(module.multiplier, (int, float)) or module.multiplier != multiplier
                ):
                    raise ValueError("FlowCPO requires fixed policy multiplier 1 and inactive EMA multiplier 0")
        if any(module.training for module in self.ema_network.modules()):
            raise ValueError("FlowCPO EMA must stay in eval mode outside old inference")
        if any(parameter.requires_grad for parameter in self.ema_network.parameters()):
            raise ValueError("FlowCPO EMA matrices must remain frozen")

    def on_train_start(self, args, accelerator, network, transformer, optimizer):
        # Resume already ran. Never initialize or copy policy into EMA here.
        super().on_train_start(args, accelerator, network, transformer, optimizer)
        policy = accelerator.unwrap_model(network)
        self._check_policy_mode(policy)
        _reject_optimizer_mode_swaps(optimizer)
        policy_ids = {id(parameter) for parameter in policy.parameters()}
        if any(id(parameter) not in policy_ids for group in optimizer.param_groups for parameter in group["params"]):
            raise ValueError("FlowCPO optimizer must own only incremental policy parameters, never base, stage one or EMA")
        if any(parameter.dtype != torch.float32 for parameter in policy.parameters()):
            raise ValueError("FlowCPO policy matrices must remain FP32")

    def on_post_optimizer_step(self, args, accelerator, network, transformer, sync_gradients, global_step):
        # Called inside accelerator.accumulate immediately after step/scheduler/
        # zero_grad. The real Accelerator overflow flag is still valid here;
        # sync_gradients alone does NOT imply a genuine mixed-precision update.
        if not sync_gradients or not accelerator.sync_gradients or accelerator.optimizer_step_was_skipped:
            return
        policy = accelerator.unwrap_model(network)
        self._check_policy_mode(policy)
        self.ema.update(policy)

    def _register_hooks_and_resume(self, args, accelerator, network):
        self._require_ema()
        if self.reference_contract is None:
            raise RuntimeError("FlowCPO provenance must be captured before resume")

        def save_ema(models, weights, output_dir):
            if accelerator.is_main_process:
                self.ema.save(output_dir)

        def load_ema(models, input_dir):
            # This is the FIRST load pre-hook, before model-list mutation or
            # model/optimizer state reads. Validate provenance before EMA mutation.
            validate_resume_contract(input_dir, self.reference_contract)
            self.ema.load(input_dir)

        accelerator.register_save_state_pre_hook(save_ema)
        accelerator.register_load_state_pre_hook(load_ema)
        # Parent registers reference provenance then ordinary policy-only hooks,
        # and finally triggers Accelerator.load_state if --resume was specified.
        super()._register_hooks_and_resume(args, accelerator, network)

    def extra_metadata(self, args):
        metadata = super().extra_metadata(args)
        metadata.update(
            ss_flow_cpo_beta=str(args.flow_cpo_beta),
            ss_flow_cpo_lambda=str(args.flow_cpo_lambda),
            ss_flow_cpo_ema_decay=str(args.flow_cpo_ema_decay),
            ss_flow_cpo_objective=self.reference_contract["objective"],
        )
        return metadata

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
        if args.post_training != "flow_cpo":
            raise ValueError("Krea2FlowCPOTrainer requires flow_cpo batches")
        self._freeze_reference()
        policy_network = accelerator.unwrap_model(network)
        self._check_policy_mode(policy_network)
        pairs, t, combined, pair_batch, paired_noise, noisy, timesteps = self._prepare_paired_batch(
            args, accelerator, batch, latents, noise, dit_dtype, network_dtype
        )
        pair_batch["_krea2_pair_count"] = pairs
        with old_adapter_context(policy_network, self.ema_network, accelerator.unwrap_model(transformer)):
            old = self.call_dit(args, accelerator, transformer, combined, pair_batch, paired_noise, noisy, timesteps, network_dtype)
        # No adapter state mutation follows this forward until checkpointed
        # backward completes and a genuine optimizer update has occurred.
        policy = self.call_dit(args, accelerator, transformer, combined, pair_batch, paired_noise, noisy, timesteps, network_dtype)
        loss, metrics = flow_cpo_loss(
            policy.pred[:pairs],
            policy.pred[pairs:],
            old.pred[:pairs],
            old.pred[pairs:],
            policy.target[:pairs],
            policy.target[pairs:],
            args.flow_cpo_beta,
            args.flow_cpo_lambda,
        )
        metrics["flow_cpo/timestep_mean"] = t.mean().item()
        metrics["flow_cpo/ema_updates"] = self.ema_updates
        return loss, metrics
